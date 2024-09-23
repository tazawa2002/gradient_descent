#include <iostream>
#include <fstream>
#include <cmath>
#include "optimizer.h"

// 勾配降下法のパラメータ
double learn_rate = 0.001;  // 学習率
int max_steps = 10000;         // 最大ステップ数

double rosenbrock(double x, double y);
void heatmap(double x_min, double x_max, double y_min, double y_max, std::function<double(double, double)> function);
void save_data(std::ofstream &data_file, int step, double x, double y, double f);

int main() {
    double x = -1.5; // -1.5; // 初期値 x
    double y = 1.5; // 1.5;  // 初期値 y

    std::function<double(double, double)> function = rosenbrock;

    // 最適化手法の選択
    std::cout << "最適化手法を選択してください:" << std::endl;
    std::cout << "1: Nomal" << std::endl;
    std::cout << "2: Momentum" << std::endl;
    std::cout << "3: Nesterov" << std::endl;
    std::cout << "4: AdaGrad" << std::endl;
    std::cout << "5: RMSprop" << std::endl;
    std::cout << "6: AdaDelta" << std::endl;
    std::cout << "7: Adam" << std::endl;
    std::cout << "8: Metropolis" << std::endl;
    std::cout << "9: NewtonRaphson" << std::endl;

    int choice;
    std::cin >> choice;

    Optimizer* optimizer = nullptr;

    switch (choice) {
        case 1:
            optimizer = new NormalOptimizer(learn_rate);
            break;
        case 2:
            optimizer = new MomentumOptimizer(learn_rate, 0.9);
            break;
        case 3:
            optimizer = new NesterovOptimizer(learn_rate, 0.9, function);
            break;
        case 4:
            optimizer = new AdaGradOptimizer(0.9);
            break;
        case 5:
            optimizer = new RMSpropOptimizer(learn_rate, 0.9);
            break;
        case 6:
            optimizer = new AdaDeltaOptimizer(0.9);
            break;
        case 7:
            optimizer = new AdamOptimizer(learn_rate, 0.9, 0.999);
            break;
        case 8:
            optimizer = new MetropolisOptimizer(20, 0.99, function);
            break;
        case 9:
            optimizer = new NewtonRaphsonOptimizer(function);
            break;
        default:
            std::cerr << "無効な選択です。" << std::endl;
            return 1;
    }

    std::ofstream data_file("gradient_descent.dat");
    if (!data_file) {
        std::cerr << "ファイルが開けませんでした。" << std::endl;
        return 1;
    }

    // 初期値の書き込み
    save_data(data_file, 0, x, y, function(x, y));
    

    // 勾配降下法のループ
    for (int step = 1; step <= max_steps; ++step) {
        double grad_x, grad_y;
        gradient(x, y, function, grad_x, grad_y);
        // stochastic_gradient(x, y, function, grad_x, grad_y);
        optimizer->update(x, y, grad_x, grad_y); // 更新関数の呼び出し
        save_data(data_file, step, x, y, function(x, y)); // 更新後の値を保存

        // 収束判定 (勾配が小さい場合に停止)
        gradient(x, y, function, grad_x, grad_y);
        if (std::abs(grad_x) < 1e-6 && std::abs(grad_y) < 1e-6) {
            break;
        }
    }

    data_file.close();
    delete optimizer;
    std::cout << "データの書き込みが完了しました。" << std::endl;

    heatmap(-2, 2, -1, 3, function);

    return 0;
}

// Rosenbrock関数 f(x, y)
double rosenbrock(double x, double y) {
    return pow(1 - x, 2) + 100 * pow(y - x * x, 2);
}

void heatmap(double x_min, double x_max, double y_min, double y_max, std::function<double(double, double)> function) {
    std::ofstream heatmap_file("heatmap.dat");

    if (!heatmap_file) {
        std::cerr << "ファイルが開けませんでした。" << std::endl;
    }

    // x, yの範囲でヒートマップのデータを作成
    for (double x = x_min; x <= x_max; x += 0.05) {
        for (double y = y_min; y <= y_max; y += 0.05) {
            heatmap_file << x << " " << y << " " << function(x, y) << std::endl;
        }
        heatmap_file << std::endl; // Gnuplot用の行区切り
    }

    heatmap_file.close();
    std::cout << "ヒートマップデータの書き込みが完了しました。" << std::endl;
}

// 更新したx, yの値をファイルに保存する関数
void save_data(std::ofstream &data_file, int step, double x, double y, double f) {
    data_file << step << " " << x << " " << y << " " << f << std::endl;
}