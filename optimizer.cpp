/**
 * @file optimizer.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "optimizer.h"
#include <functional>

// 数値微分を用いてRosenbrock関数の勾配 ∇f(x, y)を計算する関数
void gradient(double x, double y, std::function<double(double, double)> function, double &grad_x, double &grad_y) {
    double epsilon = 1e-6;
    grad_x = (function(x + epsilon, y) - function(x - epsilon, y)) / (2 * epsilon); // x方向の勾配
    grad_y = (function(x, y + epsilon) - function(x, y - epsilon)) / (2 * epsilon); // y方向の勾配
}

double gradient_x(double x, double y, std::function<double(double, double)> function) {
    double epsilon = 1e-6;
    return (function(x + epsilon, y) - function(x - epsilon, y)) / (2 * epsilon);
}

double gradient_y(double x, double y, std::function<double(double, double)> function) {
    double epsilon = 1e-6;
    return (function(x, y + epsilon) - function(x, y - epsilon)) / (2 * epsilon);
}

// 数値微分を用いてRosenbrock関数の勾配 ∇f(x, y)を計算する関数
void stochastic_gradient(double x, double y, std::function<double(double, double)> function, double &grad_x, double &grad_y) {
    double epsilon = 1e-6;
    static std::default_random_engine rng(std::random_device{}());
    std::normal_distribution<double> distribution(0, 0.05);
    double x_hat = x + distribution(rng);
    double y_hat = y + distribution(rng);
    grad_x = (function(x_hat + epsilon, y_hat) - function(x_hat - epsilon, y_hat)) / (2 * epsilon); // x方向の勾配
    grad_y = (function(x_hat, y_hat + epsilon) - function(x_hat, y_hat - epsilon)) / (2 * epsilon); // y方向の勾配
}

void NormalOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    x -= learning_rate * grad_x; // xの更新
    y -= learning_rate * grad_y; // yの更新
}

void MomentumOptimizer::update(double &x, double &y, double grad_x, double grad_y) {

    // 速度の更新
    vel_x = momentum * vel_x - learning_rate * grad_x; // ネステロフの更新
    vel_y = momentum * vel_y - learning_rate * grad_y; // ネステロフの更新

    // x, yの更新
    x += vel_x;
    y += vel_y;
}

void NesterovOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    // 先に仮の位置を計算
    double temp_x = x + momentum * vel_x;
    double temp_y = y + momentum * vel_y;

    // 勾配を計算
    double grad_temp_x, grad_temp_y;
    gradient(temp_x, temp_y, function, grad_temp_x, grad_temp_y);

    // 速度の更新
    vel_x = momentum * vel_x - learning_rate * grad_temp_x; // ネステロフの更新
    vel_y = momentum * vel_y - learning_rate * grad_temp_y; // ネステロフの更新

    // x, yの更新
    x += vel_x;
    y += vel_y;
}

void AdaGradOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    // 勾配の二乗を累積
    accumulated_grad_x += grad_x * grad_x;
    accumulated_grad_y += grad_y * grad_y;

    // 学習率の調整
    double adjusted_lr_x = learning_rate / (std::sqrt(accumulated_grad_x) + 1e-8); // 小さな値を加算してゼロ除算を防ぐ
    double adjusted_lr_y = learning_rate / (std::sqrt(accumulated_grad_y) + 1e-8); // 小さな値を加算してゼロ除算を防ぐ

    // x, yの更新
    x -= adjusted_lr_x * grad_x;
    y -= adjusted_lr_y * grad_y;
}

void RMSpropOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    // 勾配の二乗を累積
    accumulated_grad_x = decay_rate * accumulated_grad_x + (1 - decay_rate) * grad_x * grad_x;
    accumulated_grad_y = decay_rate * accumulated_grad_y + (1 - decay_rate) * grad_y * grad_y;

    // 学習率の調整
    double adjusted_lr_x = learning_rate / (sqrt(accumulated_grad_x) + 1e-8); // 小さな値を加算してゼロ除算を防ぐ
    double adjusted_lr_y = learning_rate / (sqrt(accumulated_grad_y) + 1e-8); // 小さな値を加算してゼロ除算を防ぐ

    // x, yの更新
    x -= adjusted_lr_x * grad_x;
    y -= adjusted_lr_y * grad_y;
}

void AdaDeltaOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    // 勾配の二乗を累積
    accumulated_grad_x = decay_rate * accumulated_grad_x + (1 - decay_rate) * grad_x * grad_x;
    accumulated_grad_y = decay_rate * accumulated_grad_y + (1 - decay_rate) * grad_y * grad_y;

    // 更新量の計算
    double update_x = sqrt(delta_x + 1e-8) / sqrt(accumulated_grad_x + 1e-8) * grad_x;
    double update_y = sqrt(delta_y + 1e-8) / sqrt(accumulated_grad_y + 1e-8) * grad_y;

    // x, yの更新
    x -= update_x;
    y -= update_y;

    // 更新量の累積
    delta_x = decay_rate * delta_x + (1 - decay_rate) * update_x * update_x;
    delta_y = decay_rate * delta_y + (1 - decay_rate) * update_y * update_y;
}

void AdamOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    t++; // 時間ステップの更新

    // 一次モーメントの更新
    m_x = beta1 * m_x + (1 - beta1) * grad_x;
    m_y = beta1 * m_y + (1 - beta1) * grad_y;

    // 二次モーメントの更新
    v_x = beta2 * v_x + (1 - beta2) * grad_x * grad_x;
    v_y = beta2 * v_y + (1 - beta2) * grad_y * grad_y;

    // バイアス補正
    double m_x_hat = m_x / (1 - pow(beta1, t));
    double m_y_hat = m_y / (1 - pow(beta1, t));
    double v_x_hat = v_x / (1 - pow(beta2, t));
    double v_y_hat = v_y / (1 - pow(beta2, t));

    // x, yの更新
    x -= learning_rate * m_x_hat / (sqrt(v_x_hat) + 1e-8);
    y -= learning_rate * m_y_hat / (sqrt(v_y_hat) + 1e-8);
}

void MetropolisOptimizer::update(double &x, double &y, double grad_x, double grad_y) {
    double current_energy = energy_function(x, y);
    
    // 新しい候補点を生成
    double new_x = x + generate_normal_random(0.0, 0.1); // -2.0から2.0の範囲でランダム
    double new_y = y + generate_normal_random(0.0, 0.1); // -1.0から3.0の範囲でランダム
    double new_energy = energy_function(new_x, new_y);

    // エネルギー差を計算
    double delta_energy = new_energy - current_energy;

    // メトロポリス基準に従って受け入れるか決定
    if (delta_energy < 0 || (exp(-delta_energy / temperature) > ((double) rand() / RAND_MAX))) {
        x = new_x;
        y = new_y;
    }

    temperature *= decay_rate;
}

// 正規分布に基づく乱数生成関数
double MetropolisOptimizer::generate_normal_random(double mean, double stddev) {
    // 正規分布に基づいて新しい候補点を生成
    std::normal_distribution<double> distribution(mean, stddev); // 平均0、標準偏差0.1の正規分布
    return distribution(rng);
}

void NewtonRaphsonOptimizer::update(double &x, double &y, double grad_x, double grad_y) {

    // 勾配を計算
    double gradient_x = calc_gradient_x(x, y, function);
    double gradient_y = calc_gradient_y(x, y, function);

    // ヘッセ行列を計算
    double hessian_xx = calc_hessian_xx(x, y, function); // ∂²f/∂x²
    double hessian_xy = calc_hessian_xy(x, y, function); // ∂²f/∂x∂y
    double hessian_yx = calc_hessian_yx(x, y, function); // ∂²f/∂y∂x
    double hessian_yy = calc_hessian_yy(y, x, function); // ∂²f/∂y²

    // ヘッセ行列の逆行列を計算（2x2の場合）
    double det = hessian_xx * hessian_yy - hessian_xy * hessian_yx;
    if (det == 0) {
        std::cerr << "Hessian matrix is singular!" << std::endl;
        return;
    }

    // ヘッセ行列の逆行列を用いて次の点を計算
    double inv_hessian_xx = hessian_yy / det;
    double inv_hessian_xy = -hessian_xy / det;
    double inv_hessian_yx = -hessian_yx / det;
    double inv_hessian_yy = hessian_xx / det;

    // 更新式
    double delta_x = -(inv_hessian_xx * gradient_x + inv_hessian_xy * gradient_y);
    double delta_y = -(inv_hessian_yx * gradient_x + inv_hessian_yy * gradient_y);

    // 新しい点を計算
    x += delta_x;
    y += delta_y;
}

// 数値微分を用いて勾配を計算
double NewtonRaphsonOptimizer::calc_gradient_x(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (func(x + epsilon, y) - func(x - epsilon, y)) / (2 * epsilon);
}

double NewtonRaphsonOptimizer::calc_gradient_y(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (func(x, y + epsilon) - func(x, y - epsilon)) / (2 * epsilon);
}

// ヘッセ行列の要素を数値微分で計算
double NewtonRaphsonOptimizer::calc_hessian_xx(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (calc_gradient_x(x + epsilon, y, func) - calc_gradient_x(x - epsilon, y, func)) / (2 * epsilon);
}

double NewtonRaphsonOptimizer::calc_hessian_yy(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (calc_gradient_y(x, y + epsilon, func) - calc_gradient_y(x, y - epsilon, func)) / (2 * epsilon);
}

double NewtonRaphsonOptimizer::calc_hessian_xy(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (calc_gradient_y(x + epsilon, y, func) - calc_gradient_y(x - epsilon, y, func)) / (2 * epsilon);
}

double NewtonRaphsonOptimizer::calc_hessian_yx(double x, double y, std::function<double(double, double)> func) {
    double epsilon = 1e-6;
    return (calc_gradient_x(x, y + epsilon, func) - calc_gradient_x(x, y - epsilon, func)) / (2 * epsilon);
}