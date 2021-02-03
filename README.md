# 1 BP神经网络实现的原理
 ### 1.1目标：实现异或检测，即
 
								（1，1，0）=（0）;
								（0，1，1）=（0）;
								（1，0，1）=（0）;	
								（1，0，0）=（1）;	
								（0，1，0）=（1）;	
								（0，0，1）=（1）;	
								（0，0，0）=（0）;	
								（1，1，1）=（1）;							
#### 结构：输入层->隐层->隐层->隐层->输出层
### 1.2实现过程
***
#### a.让数据完成前向传播，如：让（1，1，0）作为输入层结点的输入数据，让输入到隐含层的一个结点的数据经过一次加权和``o1=(1*w1.1+1*w1.2+0*w1.3)``，接着加权和经过一次激活函数``sigmod x1=(1.0 / (1 + exp(-o1-b1.1))``得出激活值``x1``，激活值将用于对下一层的输入也即这一层的输出。
#### b.让数据反向传播，目的是更新权重、偏置。如：让我们实际输出值0与上述完成正向传播后的值x4来进行偏差计算 ``q = (0-x4)*x4*(1-x4)``,更新权重``w4``,学习率``rate``设为0.9，``w4=w4+rate*q*x3``,更新偏置``b4=b4+rate*q``,将所有的权重和偏置更新完即完成一次反向传播。
#### c.设置样本数据给予神经网络训练，训练完成后给予测试数据查看训练效果。
# 2 BP神经网络
### 三个输入 三个隐含层 一个输出
### 1.1 定义结点、权重、偏置、误差、学习率
***
```c
#define innode 3//输入层结点数
#define hidenode 10//隐层结点数
#define outnode 1//输出层结点数
#define trainsample 75//训练样本数
#define testsample 75//测试样本数

double trainData[trainsample][innode];//输入样本
double outData[trainsample][outnode];//输出样本

double testData[testsample][innode];//测试样本

double w1[innode][hidenode];//输入层到隐层的权值
double w2[hidenode][hidenode];//隐层到隐层的权值
double w3[hidenode][hidenode];//隐层到隐层的权值
double w4[hidenode][outnode];//隐层到输出层的权值

double b1[hidenode];//隐层阈值
double b2[hidenode];//隐层阈值
double b3[hidenode];//隐层阈值
double b4[outnode];//输出层阈值

double e = 0.0;//误差计算
double error = 1.0;//允许的最大误差
double rate = 0.9;//学习率
```
***
### 1.2 初始化权重和偏置（阈值）
***
````c
//初始化函数（0到1之间的数）
void init(double w[], int n)
{
    int i;
    srand((unsigned int)time(NULL));    //这个是种子函数srand（）为rand函数提供不同的种子，每次运行程序产生不同的随机数，不然rand函数每次运行程序产生的随机数都是一样的
    for (i = 0; i < n; i++)
    {
        w[i] = 2.0 * ((double)rand() / RAND_MAX) - 1;     //RAND_MAX 是 <stdlib.h> 中伪随机数生成函数 rand 所能返回的最大数值。
                                                                                 //这意味着，任何一次对 rand 的调用，都将得到一个 0~RAND_MAX 之间的伪随机数。RAND_MAX=0x7fff
    }
}
//对权值和阈值进行初始化
    init((double*)w1, innode * hidenode);       //double *表示指向double型的指针，w为输入层到隐层的权值数组，innode*hidenode=4*10=40
    init((double*)w2, hidenode * hidenode);
    init((double*)w3, hidenode * hidenode);
    init((double*)w4, hidenode * outnode);
    init(b1, hidenode);    //b1为隐层阈值，hidenode为隐层结点数
    init(b2, hidenode);
    init(b3, hidenode);
    init(b4, outnode);	  //b4为输出层阈值，outnode为输出层结点数
````
***
### 1.3 输入数据正向传播
***
````c
	double x[innode];//输入层的输入值
    double yd[outnode];//期望的输出值

    double o1[hidenode];//隐层1结点激活值
    double o3[hidenode];//隐层2结点激活值
    double o2[hidenode];//隐层3结点激活值
    double o4[hidenode];//输出层结点激活值

    double x1[hidenode];//隐层1向输出层的输入
    double x2[hidenode];//隐层2向输出层的输入
    double x3[hidenode];//隐层3向输出层的输入
    double x4[outnode];//输出结点的输出
    /*********************************************************************
    o1: 隐层各结点的激活值等于与该结点相连的各路径上权值与该路径上的输入相乘后全部相加
    **********************************************************************
    x1: 隐层结点的输出，计算出o1后才能计算x1，等于 1.0/(1.0 + exp-(激活值+该结点的阈值))
    ***********************************************************************
    o4: 输出层结点的激活值等于与该结点相连的各路径上的权值与该路径的输入相乘后全部相加
    ***********************************************************************
    x4: 输出层结点的输出，计算出o2后才能计算x2，等于 1.0/(1.0 + exp-(激活值+该结点的阈值))
    ***********************************************************************/
    int issamp;
    int i, j, k;
    for (issamp = 0; issamp < trainsample; issamp++)
    {
        for (i = 0; i < innode; i++)
            x[i] = trainData[issamp][i];//赋予输入值

        for (i = 0; i < outnode; i++)
            yd[i] = label[issamp][i];//赋予期望值

        //（输入层到隐层1）计算隐层各结点的激活值o和隐层的输出值x
        for (i = 0; i < hidenode; i++)
        {
            o1[i] = 0.0;//激活值
            for (j = 0; j < innode; j++)
                o1[i] = o1[i] + w1[j][i] * x[j];              //w[][]为输入层到隐含层的网络权重，o1是隐层结点激活值（算出加权和）
            x1[i] = 1.0 / (1.0 + exp(-o1[i] - b1[i]));  //x1是隐含层的输出
        }

        //隐层1到2
        for (i = 0; i < hidenode; i++)
        {
            o2[i] = 0.0;//激活值
            for (j = 0; j < hidenode; j++)
                o2[i] = o2[i] + w2[j][i] * x1[j];              //w[][]为输入层到隐含层的网络权重，o1是隐层结点激活值（算出加权和）
            x2[i] = 1.0 / (1.0 + exp(-o2[i] - b2[i]));  //x2是隐含层的输出
        }

        //隐层2到3
        for (i = 0; i < hidenode; i++)
        {
            o3[i] = 0.0;//激活值
            for (j = 0; j < hidenode; j++)
                o3[i] = o3[i] + w3[j][i] * x2[j];              //w[][]为输入层到隐含层的网络权重，o1是隐层结点激活值（算出加权和）
            x3[i] = 1.0 / (1.0 + exp(-o3[i] - b3[i]));  //x3是隐含层的输出
        }


        //（隐层3到输出）计算输出层各结点的激活值和输出值
        for (i = 0; i < outnode; i++)
        {
            o4[i] = 0.0;
            for (j = 0; j < hidenode; j++)
                o4[i] = o4[i] + w4[j][i] * x3[j];                //w1[][]为隐层到输出层的权值，o2为输出层结点激活值
            x4[i] = 1.0 / (1.0 + exp(-o4[i] - b4[i]));        //x4为输出结点的输出
        }
        //正向传播完成
        //得到了x4输出后接下来就要进行反向传播了
````
***
### 1.4 开始反向传播
***
````c
    /*从前往后——输出单元偏差(误差)qq计算方式：  （期望输出 - 实际输出）乘上 实际输出 乘上 （1-实际输出）  */
    double qq[outnode];//期望的输出与实际输出的偏差
    double pp1[hidenode];//（隐1）隐含结点校正误差
    double pp2[hidenode];//隐3到隐2
    double pp3[hidenode];//输出层到隐3
````
#### 更新权重（权值）
````c
//计算实际输出与期望输出的偏差，反向调节隐层到输出层的路径上的权值
        for (i = 0; i < outnode; i++)
        {
            qq[i] = (yd[i] - x4[i]) * x4[i] * (1 - x4[i]);      //输出节点j的偏差，yd是已经获取的样本期望值
            for (j = 0; j < hidenode; j++)
                w4[j][i] = w4[j][i] + rate_w1 * qq[i] * x3[j];   //隐层到输出层的路径上的权值
        }   

        //隐3到隐2
        for (i = 0; i < hidenode; i++)
        {
            pp3[i] = 0.0;
            for (j = 0; j < outnode; j++)
                pp3[i] = pp3[i] + qq[j] * w4[i][j];
            pp3[i] = pp3[i] * x3[i] * (1.0 - x3[i]);
            for (k = 0; k < hidenode; k++)
                w3[k][i] = w3[k][i] + rate_w3 * pp3[i] * x2[k];
        }
            
        //隐2到隐1
        for (i = 0; i < hidenode; i++)
        {
            pp2[i] = 0.0;
            for (k = 0; k < hidenode; k++)
                pp2[i] = pp2[i] + pp3[j] * w3[i][j];
            pp2[i] = pp2[i] * x2[i] * (1.0 - x2[i]);
            for (k = 0; k < hidenode; k++)
                w2[k][i] = w2[k][i] + rate_w2 * pp2[i] * x1[k];
        }

        //继续反向传播调整输入层到隐1层的各路径上的权值
        for (i = 0; i < hidenode; i++)
        {
            pp1[i] = 0.0;
            for (j = 0; j < hidenode; j++)
                pp1[i] = pp1[i] + pp2[j] * w2[i][j];
            pp1[i] = pp1[i] * x1[i] * (1.0 - x1[i]);      //隐含层节点i的偏差

            for (k = 0; k < innode; k++)
                w1[k][i] = w1[k][i] + rate_w1 * pp1[i] * x[k];  //输入层到隐层的各路径上的权值
        }
````
#### 更新偏置（阈值）
````c
        //调整各结点的阈值
        for (k = 0; k < outnode; k++)
            b4[k] = b4[k] + rate_b2 * qq[k];
        for (k = 0; k < hidenode; k++)
            b3[k] = b3[k] + rate_b4 * pp3[k];
        for (k = 0; k < hidenode; k++)
            b2[k] = b2[k] + rate_b3 * pp2[k];
        for (j = 0; j < hidenode; j++)
            b1[j] = b1[j] + rate_b1 * pp1[j];
````
#### 调整误差
````c
        //调整允许的最大误差
        for (k = 0; k < outnode; k++)
        {
            e += fabs(yd[k] - x4[k]) * fabs(yd[k] - x4[k]); //计算均方差  
        }
        error = e / 2.0;   
````
***
### 1.5 训练完成后开始输入样本
***
##### 相当于做正向传播（用训练更新后的权值和阈值）
````c
//Bp识别
double* recognize(double* p)
{
    double x[innode];//输入层的输入值
    double o1[hidenode];//隐层1结点激活值
    double o2[hidenode];//隐层2结点激活值
    double o3[hidenode];//隐层3结点激活值
    double o4[hidenode];//输出层结点激活值

    double x1[hidenode];//隐层1向输出层的输入
    double x2[hidenode];//隐层2向输出层的输入
    double x3[hidenode];//隐层3向输出层的输入
    double x4[outnode];//输出结点的输出
    int i, j, k;

    for (i = 0; i < innode; i++)
        x[i] = p[i];

    for (j = 0; j < hidenode; j++)
    {
        o1[j] = 0.0;
        for (i = 0; i < innode; i++)
            o1[j] = o1[j] + w1[i][j] * x[i]; //隐含层各单元激活值  
        x1[j] = 1.0 / (1.0 + exp(-o1[j] - b1[j])); //隐含层各单元输出  
    }

    for (j = 0; j < hidenode; j++)
    {
        o2[j] = 0.0;
        for (i = 0; i < hidenode; i++)
            o2[j] = o2[j] + w2[i][j] * x1[i]; //隐含层各单元激活值  
        x2[j] = 1.0 / (1.0 + exp(-o2[j] - b2[j])); //隐含层各单元输出  
    }

    for (j = 0; j < hidenode; j++)
    {
        o3[j] = 0.0;
        for (i = 0; i < hidenode; i++)
            o3[j] = o3[j] + w3[i][j] * x2[i]; //隐含层各单元激活值  
        x3[j] = 1.0 / (1.0 + exp(-o3[j] - b3[j])); //隐含层各单元输出  
    }

    for (k = 0; k < outnode; k++)
    {
        o4[k] = 0.0;
        for (j = 0; j < hidenode; j++)
            o4[k] = o4[k] + w4[j][k] * x3[j];//输出层各单元激活值  
        x4[k] = 1.0 / (1.0 + exp(-o4[k] - b4[k]));//输出层各单元输出   
    }

    for (k = 0; k < outnode; k++)
    {
        result[k] = x4[k];
    }
    return result;//返回结果
}
````
***
