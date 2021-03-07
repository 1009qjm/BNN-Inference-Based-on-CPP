#include <stdio.h>
#include <windows.h>
#include<iostream>
#include<stdlib.h>
#include<cmath>
#pragma GCC optimize(3,"Ofast","inline")
using namespace std;

float img[10000][28][28];
int label[10000];

inline int xnor(int a,int b){
    return (a==b)?1:0;
}

void bconv(int ch_in,int ch_out,int pad,int stride,int k,int h,int w,int *in,int *weight,int *out){
    int h_o,w_o;
    int i,j,n,m;
    int kx,ky;
    h_o=(h-k+2*pad)/stride+1;
    w_o=(w-k+2*pad)/stride+1;
    for(i=0;i<h_o;i++)
        for(j=0;j<w_o;j++)
            for(m=0;m<ch_out;m++){
                //计算out[m][i][j]的地址
                int addr_o=m*h_o*w_o+i*w_o+j;
                *(out+addr_o)=0;
                for(n=0;n<ch_in;n++)
                    for(kx=0;kx<k;kx++)
                        for(ky=0;ky<k;ky++)
                    {
                        int row=i*stride+kx-pad;
                        int col=j*stride+ky-pad;
                        int addr_i=n*h*w+row*w+col;
                        //
                        int addr_w=m*ch_in*k*k+n*k*k+kx*k+ky;
                        int a=0;
                        int b=*(weight+addr_w);
                        if(row>=0&&row<h&&col>=0&&col<w)                 //内部
                            a=*(in+addr_i);
                        
                        *(out+addr_o)+=xnor(a,b);
                    }
            }
}

void conv(int ch_in,int ch_out,int pad,int stride,int k,int h,int w,float* in,float *weight,float *bias,float *out){
    int h_o,w_o;
    int i,j,n,m;
    int kx,ky;
    h_o=(h-k+2*pad)/stride+1;
    w_o=(w-k+2*pad)/stride+1;
    for(i=0;i<h_o;i++)
        for(j=0;j<w_o;j++)
            for(m=0;m<ch_out;m++){
                //计算out[m][i][j]的地址
                int addr_o=m*h_o*w_o+i*w_o+j;
                *(out+addr_o)=bias[m];
                for(n=0;n<ch_in;n++)
                    for(kx=0;kx<k;kx++)
                        for(ky=0;ky<k;ky++)
                    {
                        //out[m][i][j]+=in[n][i*stride+kx-pad][j*stride+ky-pad]*weight[m][n][kx][ky]
                        int row=i*stride+kx-pad;
                        int col=j*stride+ky-pad;
                        int addr_i=n*h*w+row*w+col;
                        //
                        int addr_w=m*ch_in*k*k+n*k*k+kx*k+ky;
                        if(row>=0&&row<h&&col>=0&&col<w)
                            *(out+addr_o)+=(*(in+addr_i))*(*(weight+addr_w));
                    }
    }
}

void pool(int h,int w,int ch,float* in,float* out){
    int i,j,n,kx,ky;
    for(i=0;i<h/2;i++)
        for(j=0;j<w/2;j++)
            for(n=0;n<ch;n++){
                //float tmp=in[n][2*i][2*j]
                float tmp1=*(in+n*h*w+2*i*w+2*j);
                float tmp2=*(in+n*h*w+2*i*w+2*j+1);
                float tmp3=*(in+n*h*w+(2*i+1)*w+2*j);
                float tmp4=*(in+n*h*w+(2*i+1)*w+2*j+1);
                float max1=(tmp1>tmp2)?tmp1:tmp2;
                float max2=(tmp3>tmp4)?tmp3:tmp4;
                *(out+n*h*w/4+i*w/2+j)=(max1>max2)?max1:max2;
    }
}

void pool(int h,int w,int ch,int* in,int* out){
    int i,j,n,kx,ky;
    for(i=0;i<h/2;i++)
        for(j=0;j<w/2;j++)
            for(n=0;n<ch;n++){
                //float tmp=in[n][2*i][2*j]
                int tmp1=*(in+n*h*w+2*i*w+2*j);
                int tmp2=*(in+n*h*w+2*i*w+2*j+1);
                int tmp3=*(in+n*h*w+(2*i+1)*w+2*j);
                int tmp4=*(in+n*h*w+(2*i+1)*w+2*j+1);
                int max1=(tmp1>tmp2)?tmp1:tmp2;
                int max2=(tmp3>tmp4)?tmp3:tmp4;
                *(out+n*h*w/4+i*w/2+j)=(max1>max2)?max1:max2;
    }
}

void global_avg_pool(int* in,int h,int w,int ch,float* out){
    for(int c=0;c<ch;c++){
        int tmp=0;
        for(int i=0;i<w;i++)
            for(int j=0;j<h;j++)
                tmp+=*(in+c*h*w+i*w+j);
        out[c]=(float)tmp/(h*w);
    }
}

void read_param(float* buff,int length,const char* filename){
    FILE *fp;
    if((fp=fopen(filename,"rb"))==NULL){
        printf("Fail to open file!\n");
        system("pause");
        exit(0);
    }
    fread(buff,sizeof(float),length,fp);
    fclose(fp);
}

void read_param_int(int* buff,int length,const char* filename){
    FILE *fp;
    float *tmp=new float[length];
    if((fp=fopen(filename,"rb"))==NULL){
        printf("Fail to open file!\n");
        exit(0);
    }
    fread(tmp,sizeof(float),length,fp);
    for(int i=0;i<length;i++)
        buff[i]=(int)tmp[i];
    fclose(fp);
}

void bn(float* gamma,float *beta,float *mean,float *var,int channel,int h,int w,float* in,float *out){
    for(int ch=0;ch<channel;ch++)
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
                {
                    *(out+ch*h*w+i*w+j)=(*(in+ch*h*w+i*w+j)-mean[ch])*gamma[ch]/sqrt(var[ch]+0.00001)+beta[ch];
                }
    return;
}

void relu(float* in,float* out,int length){
    for(int i=0;i<length;i++)
        out[i]=(in[i]>=0.0)?in[i]:0.0;
}

void relu(int* in,int* out,int length){
    for(int i=0;i<length;i++)
        out[i]=(in[i]>=0)?in[i]:0;
}

void threshold_func(int* in,int* out,int ch,int h,int w,float* S,float* T){
    for(int c=0;c<ch;c++)
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
               {
                   float tmp=(float)(*(in+c*h*w+i*w+j));
                   if((tmp>=T[c]&&S[c]>=0) || (tmp<T[c]&&S[c]<0))
                       *(out+c*h*w+i*w+j)=1;
                   else
                       *(out+c*h*w+i*w+j)=0;
               }
}

void threshold_func(float* in,int* out,int ch,int h,int w,float* S,float* T){
    for(int c=0;c<ch;c++)
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
               {
                   if((*(in+c*h*w+i*w+j)>=T[c]&&S[c]>=0) || (*(in+c*h*w+i*w+j)<T[c]&&S[c]<0))
                       *(out+c*h*w+i*w+j)=1;
                   else
                       *(out+c*h*w+i*w+j)=0;
               }
}

void affine_transform(int* in,int length,int ch,int kernel_size){
    for(int i=0;i<length;i++){
        in[i]=in[i]*2-kernel_size*kernel_size*ch;
    }
}

int main(){
    //float
    float *Wf1=new float[16*1*5*5];
    float *bf1=new float[16];
    float *mean1=new float[16];
    float *var1=new float[16];
    float *gamma1=new float[16];
    float *beta1=new float[16];
    //binary
    int *Wb2=new int[32*16*5*5];
    int *Wb3=new int[32*32*5*5];
    int *Wb4=new int[64*32*5*5];
    int *Wb5=new int[10*64*5*5];
    float *S2=new float[16];
    float *T2=new float[16];
    float *S3=new float[32];
    float *T3=new float[32];
    float *S4=new float[32];
    float *T4=new float[32];
    float *S5=new float[64];
    float *T5=new float[64];
    //read_param
    read_param(Wf1,16*1*5*5,"F:\\nn\\Wf1.bin");
    read_param(bf1,16,"F:\\nn\\bf1.bin");
    read_param_int(Wb2,32*16*5*5,"F:\\nn\\Wb2.bin");
    read_param_int(Wb3,32*32*5*5,"F:\\nn\\Wb3.bin");
    read_param_int(Wb4,64*32*5*5,"F:\\nn\\Wb4.bin");
    read_param_int(Wb5,10*64*5*5,"F:\\nn\\Wb5.bin");
    read_param(gamma1,16,"F:\\nn\\gamma1.bin");
    read_param(beta1,16,"F:\\nn\\beta1.bin");
    read_param(mean1,16,"F:\\nn\\mean1.bin");
    read_param(var1,16,"F:\\nn\\var1.bin");
    read_param(T2,16,"F:\\nn\\T2.bin");
    read_param(S2,16,"F:\\nn\\S2.bin");
    read_param(T3,32,"F:\\nn\\T3.bin");
    read_param(S3,32,"F:\\nn\\S3.bin");
    read_param(T4,32,"F:\\nn\\T4.bin");
    read_param(S4,32,"F:\\nn\\S4.bin");
    read_param(T5,64,"F:\\nn\\T5.bin");
    read_param(S5,64,"F:\\nn\\S5.bin");
    //read_data
    read_param((float*)img,10000*28*28,"F:\\nn\\image.bin");
    read_param_int(label,10000,"F:\\nn\\label.bin");
    //conv1 and pool1
    float *conv1_out=new float[16*28*28];
    float *bn1_out=new float[16*28*28];
    float *pool1_out=new float[16*14*14];
    //bconv1
    int* bconv1_in=new int[16*14*14];
    int *bconv1_out=new int[32*14*14];
    //bconv2 and pool2
    int *bconv2_in=new int[32*14*14];
    int *bconv2_out=new int[32*14*14];
    int* pool2_out=new int[32*7*7];
    //bconv3
    int *bconv3_in=new int[32*7*7];
    int *bconv3_out=new int[64*7*7];
    //bconv4
    int *bconv4_in=new int[64*7*7];
    int *bconv4_out=new int[10*7*7];
    int *relu_out=new int[10*7*7];
    //avg_pool
    float *out=new float[10];
    //
    cout<<"start inference"<<endl;
    int correct=0;
    int N=10000;
    for(int n=0;n<N;n++){
        //conv and pool
        conv(1,16,2,1,5,28,28,(float*)img[n],Wf1,bf1,conv1_out);
        bn(gamma1,beta1,mean1,var1,16,28,28,conv1_out,bn1_out);
        pool(28,28,16,bn1_out,pool1_out);
        //bconv1
        threshold_func(pool1_out,bconv1_in,16,14,14,S2,T2);
        bconv(16,32,2,1,5,14,14,bconv1_in,Wb2,bconv1_out);
        //bconv2 and pool2
        threshold_func(bconv1_out,bconv2_in,32,14,14,S3,T3);
        bconv(32,32,2,1,5,14,14,bconv2_in,Wb3,bconv2_out);
        pool(14,14,32,bconv2_out,pool2_out);
        //bconv3
        threshold_func(pool2_out,bconv3_in,32,7,7,S4,T4);
        bconv(32,64,2,1,5,7,7,bconv3_in,Wb4,bconv3_out);
        //bconv4
        threshold_func(bconv3_out,bconv4_in,64,7,7,S5,T5);
        bconv(64,10,2,1,5,7,7,bconv4_in,Wb5,bconv4_out);
        affine_transform(bconv4_out,10*7*7,64,5);
        relu(bconv4_out,relu_out,10*7*7);
        //avg_pool
        global_avg_pool(relu_out,7,7,10,out);
        //
        float max_val=-9999;
        int max_idx=-1;
        for(int i=0;i<10;i++)
            if(out[i]>max_val){
                max_val=out[i];
                max_idx=i;
            }
        if(max_idx==label[n]){
            correct++;
            cout<<"right"<<endl;
        }
        else
            cout<<"error"<<endl;
    }
    cout<<"acc is "<<(float)correct/N<<endl;
    system("pause");
    return 0;
}
