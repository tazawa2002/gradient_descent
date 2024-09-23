#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <functional>
#include <functional> 

#ifndef OPTIMIZER
#define OPTIMIZER

// 数値微分を用いてRosenbrock関数の勾配 ∇f(x, y)を計算する関数
void gradient(double x, double y, std::function<double(double, double)> function, double &grad_x, double &grad_y);
double gradient_x(double x, double y, std::function<double(double, double)> function);
double gradient_y(double x, double y, std::function<double(double, double)> function);


// 数値微分を用いてRosenbrock関数の勾配 ∇f(x, y)を計算する関数
void stochastic_gradient(double x, double y, std::function<double(double, double)> function, double &grad_x, double &grad_y);

// Optimizerのベースクラス
class Optimizer {
public:
    virtual ~Optimizer() {} // 仮想デストラクタ
    virtual void update(double &x, double &y, double grad_x, double grad_y) = 0; // 純粋仮想関数
};

// 通常の勾配法を実装するクラス
class NormalOptimizer : public Optimizer {
public:
    NormalOptimizer(double lr) : learning_rate(lr) {}

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
};

// モーメンタム法を実装するクラス
class MomentumOptimizer : public Optimizer {
public:
    MomentumOptimizer(double lr, double mom) 
        : learning_rate(lr), momentum(mom), vel_x(0.0), vel_y(0.0) {}

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
    double momentum;
    double vel_x, vel_y;
};

// モーメンタム法を実装するクラス（ネステロフの勾配加速法を含む）
class NesterovOptimizer : public Optimizer {
public:
    NesterovOptimizer(double lr, double mom, std::function<double(double, double)> function) 
        : learning_rate(lr), momentum(mom), vel_x(0.0), vel_y(0.0), function(function) {}

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
    double momentum;
    double vel_x, vel_y;
    std::function<double(double, double)> function;
};

// AdaGradを実装するクラス
class AdaGradOptimizer : public Optimizer {
public:
    AdaGradOptimizer(double lr) : learning_rate(lr) {
        accumulated_grad_x = 0.0;
        accumulated_grad_y = 0.0;
    }

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
    double accumulated_grad_x;
    double accumulated_grad_y;
};

// RMSpropを実装するクラス
class RMSpropOptimizer : public Optimizer {
public:
    RMSpropOptimizer(double lr, double decay_rate) 
        : learning_rate(lr), decay_rate(decay_rate) {
        accumulated_grad_x = 0.0;
        accumulated_grad_y = 0.0;
    }

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
    double decay_rate;
    double accumulated_grad_x;
    double accumulated_grad_y;
};

// AdaDeltaを実装するクラス
class AdaDeltaOptimizer : public Optimizer {
public:
    AdaDeltaOptimizer(double decay_rate) 
        : decay_rate(decay_rate) {
        accumulated_grad_x = 0.0;
        accumulated_grad_y = 0.0;
        delta_x = 0.0;
        delta_y = 0.0;
    }

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double decay_rate;
    double accumulated_grad_x;
    double accumulated_grad_y;
    double delta_x;
    double delta_y;
};

// Adamを実装するクラス
class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(double lr, double beta1, double beta2) 
        : learning_rate(lr), beta1(beta1), beta2(beta2) {
        m_x = 0.0;
        m_y = 0.0;
        v_x = 0.0;
        v_y = 0.0;
        t = 0;
    }

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double learning_rate;
    double beta1;
    double beta2;
    double m_x, m_y; // 一次モーメント
    double v_x, v_y; // 二次モーメント
    int t; // 時間ステップ
};

// メトロポリス法を実装するクラス
class MetropolisOptimizer : public Optimizer {
public:
    MetropolisOptimizer(double temp, double decay, std::function<double(double, double)> energy_function) : temperature(temp), decay_rate(decay), energy_function(energy_function){
        srand(static_cast<unsigned int>(time(0))); // 乱数の初期化
        rng = std::default_random_engine(std::random_device{}()); // 乱数エンジンの初期化
    }

    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    double temperature;
    double decay_rate;
    std::default_random_engine rng; // 乱数エンジン
    std::function<double(double, double)> energy_function; // エネルギー関数のポインタ

    // 正規分布に基づく乱数生成関数
    double generate_normal_random(double mean, double stddev);
};

// ニュートン・ラフソン法
class NewtonRaphsonOptimizer : public Optimizer {
public:
    NewtonRaphsonOptimizer(std::function<double(double, double)> func) : function(func){}
    void update(double &x, double &y, double grad_x, double grad_y) override;

private:
    std::function<double(double, double)> function; // 目的関数

    // 数値微分を用いて勾配を計算
    double calc_gradient_x(double x, double y, std::function<double(double, double)> func);
    double calc_gradient_y(double x, double y, std::function<double(double, double)> func);

    // ヘッセ行列の要素を数値微分で計算
    double calc_hessian_xx(double x, double y, std::function<double(double, double)> func);
    double calc_hessian_yy(double x, double y, std::function<double(double, double)> func);
    double calc_hessian_xy(double x, double y, std::function<double(double, double)> func);
    double calc_hessian_yx(double x, double y, std::function<double(double, double)> func);
};


#endif