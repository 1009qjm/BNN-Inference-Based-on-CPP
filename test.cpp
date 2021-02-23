#include <stdio.h>
#include <windows.h>
#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<ctime>
#pragma GCC optimize(3,"Ofast","inline")
using namespace std;

float img[10000][28][28];
int label[10000];

inline int xnor(int a,int b){
    //if((a!=1&&a!=0)||(b!=1&&b!=0))
        //cout<<"value error"<<endl;
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

void fold_bn(int ch,int h,int w,float *gamma,float *beta,int *in,float *out){
    for(int c=0;c<ch;c++)
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++){
                out[c*h*w+i*w+j]=(float)in[c*h*w+i*w+j]*gamma[c]+beta[c];
            }
}

void binarize(float* in,int *out,int length){
    for(int i=0;i<length;i++)
        out[i]=(in[i]>=0)?1:0;
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

void global_avg_pool(float* in,int h,int w,int ch,float* out){
    for(int c=0;c<ch;c++){
        float tmp=0.0;
        for(int i=0;i<w;i++)
            for(int j=0;j<h;j++)
                tmp+=*(in+c*h*w+i*w+j);
        out[c]=tmp/(h*w);
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
int main(){
    //float
    float *Wc1=new float[16*1*5*5];
    float *bc1=new float[16];
    float *mean=new float[16];
    float *var=new float[16];
    float *gamma=new float[16];
    float *beta=new float[16];
    //binary
    int *BWc1=new int[32*16*5*5];
    int *BWc2=new int[32*32*5*5];
    int *BWc3=new int[64*32*5*5];
    int *BWc4=new int[10*64*5*5];
    float *gamma2=new float[32];
    float *beta2=new float[32];
    float *gamma3=new float[32];
    float *beta3=new float[32];
    float *gamma4=new float[64];
    float *beta4=new float[64];
    float *gamma5=new float[10];
    float *beta5=new float[10];
    //read_param
    read_param(Wc1,16*1*5*5,"F:\\python_project\\param\\Wc1.bin");
    read_param(bc1,16,"F:\\python_project\\param\\bc1.bin");
    read_param_int(BWc1,32*16*5*5,"F:\\python_project\\param\\BWc1.bin");
    read_param_int(BWc2,32*32*5*5,"F:\\python_project\\param\\BWc2.bin");
    read_param_int(BWc3,64*32*5*5,"F:\\python_project\\param\\BWc3.bin");
    read_param_int(BWc4,10*64*5*5,"F:\\python_project\\param\\BWc4.bin");
    read_param(gamma,16,"F:\\python_project\\param\\bn1_gamma.bin");
    read_param(beta,16,"F:\\python_project\\param\\bn1_beta.bin");
    read_param(mean,16,"F:\\python_project\\param\\bn1_mean.bin");
    read_param(var,16,"F:\\python_project\\param\\bn1_var.bin");
    read_param(gamma2,32,"F:\\python_project\\param\\gamma2.bin");
    read_param(beta2,32,"F:\\python_project\\param\\beta2.bin");
    read_param(gamma3,32,"F:\\python_project\\param\\gamma3.bin");
    read_param(beta3,32,"F:\\python_project\\param\\beta3.bin");
    read_param(gamma4,64,"F:\\python_project\\param\\gamma4.bin");
    read_param(beta4,64,"F:\\python_project\\param\\beta4.bin");
    read_param(gamma5,10,"F:\\python_project\\param\\gamma5.bin");
    read_param(beta5,10,"F:\\python_project\\param\\beta5.bin");
    //read_data
    read_param((float*)img,10000*28*28,"F:\\python_project\\param\\img.bin");
    read_param_int(label,10000,"F:\\python_project\\param\\label.bin");
    //tmp variable
    //conv1 and pool1
    float *conv1_out=new float[16*28*28];
    float *bn1_out=new float[16*28*28];
    float *pool1_out=new float[16*14*14];
    //bconv1
    int* bconv1_in=new int[16*14*14];
    int *bconv1_out=new int[32*14*14];
    float *bn2_out=new float[32*14*14];
    //bconv2 and pool2
    int *bconv2_in=new int[32*14*14];
    int *bconv2_out=new int[32*14*14];
    float *bn3_out=new float[32*14*14];
    float *pool2_out=new float[32*7*7];
    //bconv3
    int *bconv3_in=new int[32*7*7];
    int *bconv3_out=new int[64*7*7];
    float *bn4_out=new float[64*7*7];
    //bconv4
    int *bconv4_in=new int[64*7*7];
    int *bconv4_out=new int[10*7*7];
    float *bn5_out=new float[10*7*7];
    float *relu_out=new float[10*7*7];
    //avg_pool
    float *out=new float[10];
    //
    cout<<"start inference"<<endl;
    clock_t start,end;
    int correct=0;
    int N=10000;
    start=clock();
    for(int n=0;n<N;n++){
        //conv and pool
        conv(1,16,2,1,5,28,28,(float*)img[n],Wc1,bc1,conv1_out);
        bn(gamma,beta,mean,var,16,28,28,conv1_out,bn1_out);
        pool(28,28,16,bn1_out,pool1_out);
        //bconv1
        binarize(pool1_out,bconv1_in,16*14*14);
        bconv(16,32,2,1,5,14,14,bconv1_in,BWc1,bconv1_out);
        fold_bn(32,14,14,gamma2,beta2,bconv1_out,bn2_out);
        //bconv2 and pool2
        binarize(bn2_out,bconv2_in,32*14*14);
        bconv(32,32,2,1,5,14,14,bconv2_in,BWc2,bconv2_out);
        fold_bn(32,14,14,gamma3,beta3,bconv2_out,bn3_out);
        pool(14,14,32,bn3_out,pool2_out);
        //bconv3
        binarize(pool2_out,bconv3_in,32*7*7);
        bconv(32,64,2,1,5,7,7,bconv3_in,BWc3,bconv3_out);
        fold_bn(64,7,7,gamma4,beta4,bconv3_out,bn4_out);
        //bconv4
        binarize(bn4_out,bconv4_in,64*7*7);
        bconv(64,10,2,1,5,7,7,bconv4_in,BWc4,bconv4_out);
        fold_bn(10,7,7,gamma5,beta5,bconv4_out,bn5_out);              //10*7*7
        relu(bn5_out,relu_out,10*7*7);
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
            //cout<<"right"<<endl;
        }
        else
            cout<<"error,n="<<n<<endl;
    }
    end=clock();
    cout<<"acc is "<<(float)correct/N<<endl;
    cout<<"time used per image is "<<(double)(end-start)/(CLK_TCK*N)<<endl;
    system("pause");
    return 0;
}
