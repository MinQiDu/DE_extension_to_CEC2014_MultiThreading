#include "DE.h"
#include <ctime>
#include <cstdlib>
#include <algorithm> /* for auto */
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>     /* for sqrt */
#include <numeric>   /* for accumulate */
#include <thread>
#include "DE_fileoutput.h" // 包含 FileOutput 函式的宣告

/* Constructor 初始化亂數引擎 */
//algo_DE::algo_DE()
//	: gen(rd()) /* 用 rd 初始化 gen（記住這只能放在初始化列表中） */
//{
//}
/* Destructor 釋放動態配置的記憶體 */
//algo_DE::~algo_DE() {
//	if (benchmark != nullptr) {
//		delete benchmark;    // 釋放動態配置的記憶體
//		benchmark = nullptr; // 好習慣，避免野指標
//	}
//}

void algo_DE::SingleRun(int r, int _dim, int _pop_size, double _CR, double _F, int _func_id,
	vector<double>& best_fit_record,
	vector<vector<double>>& eva_fit_record)
{
	int t_func_id = _func_id;
	int t_iter_c = 0;					  /* 當前迭代次數 */ 
	int t_dim = _dim;
	int r_init_pop = 18;				  /* 初始群體大小比例 */ 
	int t_pop_size = t_dim * r_init_pop;
	int t_nfes = 0;						  /* 初始化評估次數 */ 
	int t_mnfes = t_dim * 10000;			  /* 最大評估次數 */ 
	double t_CR = _CR;
	double t_F = _F;

	vector<vector<double>> t_pop;         /* dim*pop_size 個群體解 */
	vector<vector<double>> t_donor_pop;   /* 變異後的 donor 解 */
	vector<vector<double>> t_current_pop; /* crossover 後的新解 */
	vector<double> t_fit;                 /* 原解的 fitness */
	vector<double> t_current_fit;         /* 新解的 fitness */
	double t_best_fit;                    /* 紀錄最佳fitness */
	vector<double> t_best_sol;            /* 紀錄最佳解 */
	double t_lower_bound, t_upper_bound;  /* 不同函數對應的上下界 */
	int t_current_run = r;							  /* 記錄當前運行次數，供 Determination() 紀錄使用 */ 
	eva_fit_record[t_current_run].reserve(t_mnfes);   /* 為每次運行的 evaluation fitness 紀錄分配空間 */ 

	Init(
		t_dim, t_pop_size,
		t_pop, t_donor_pop, t_current_pop,
		t_fit, t_current_fit,
		t_best_fit, t_best_sol,
		t_lower_bound, t_upper_bound, 
		t_func_id
	);

	while (t_nfes < t_mnfes)
	{
		Mutation(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			t_F
		);
		Crossover(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			t_CR
		);
		Evaluation(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			t_func_id,
			t_nfes
		);
		Determination(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			eva_fit_record,
			t_current_run
		);
	}
	best_fit_record[t_current_run] = t_best_fit; // 記錄所有 run 運行的最佳 fitness
	//cout << "best fitness in run " << (r + 1) << ": " << best_fit << endl;
	/*cout << "best solution found in run " << (r + 1) << ": ";
	for (const auto& val : best_sol) {
		cout << val << " ";
	}
	cout << endl;*/

}

void algo_DE::RunALG(int _dim, int _pop_size, double _CR, double _F, int _func_id)
{
	//benchmark = new Cec14TestFunc(_dim, _func_id);  // 初始化 benchmark 函數物件
	int run = 51;									// 設定運行次數
	vector<double> best_fit_record;					// 用於記錄每次運行的最佳 fitness
	vector<vector<double>> eva_fit_record;			// 輸出每一 run 每次evaluation fitness 收斂結果用
	best_fit_record.resize(run);
	eva_fit_record.resize(run);						// 用於記錄每次運行的 evaluation fitness
	
	vector<thread> threads;
	for (int r = 0; r < run; ++r)
	{
		threads.emplace_back(&algo_DE::SingleRun, this, r, _dim, _pop_size, _CR, _F, _func_id,
			ref(best_fit_record),
			ref(eva_fit_record));
	}
	// 等待所有 thread 結束
	for (auto& t : threads) {
		t.join();
	}

	// 顯示 eva_fit_record 
	/*std::cout << eva_fit_record[current_run].size() << " evaluations recorded for run " << current_run + 1 << std::endl;
	for (const auto& fit : eva_fit_record[current_run]) {
		std::cout << fit << " ";
	}*/

	// 輸出結果，計算平均
	double avg_best_fit = accumulate(best_fit_record.begin(), best_fit_record.end(), 0.0) / run;
	std::cout << "Avg Best fitness: " << avg_best_fit << endl;
	// 輸出到檔案
	DE_FileOutput(
		run,
		best_fit_record,
		eva_fit_record,
		_func_id,
		_dim,
		_pop_size,
		_CR,
		_F,
		avg_best_fit
	);
}

void algo_DE::Init(
	int dim, int pop_size,
	vector<vector<double>>& pop, 
	vector<vector<double>>& donor_pop,
	vector<vector<double>>& current_pop, 
	vector<double>& fit,
	vector<double>& current_fit,
	double& best_fit,
	vector<double>& best_sol,
	double& lower_bound,
	double& upper_bound,
	int func_id
)
{
	// 初始化群體解
	//Benchmark benchmark(func_id);
	//benchmark.GetBounds(lower_bound, upper_bound);
	//benchmark->GetBounds(lower_bound, upper_bound);
	Cec14TestFunc benchmark(dim, func_id);
	benchmark.GetBounds(lower_bound, upper_bound);
	//cout << "Lower bound: " << lower_bound << ", Upper bound: " << upper_bound << endl;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dist_init(lower_bound, upper_bound); /* 使用從基準函數獲取的邊界 */

	pop.clear();
	current_pop.clear();
	fit.clear();
	current_fit.clear();

	pop.assign(pop_size, vector<double>(dim)); // 初始化pop
	current_pop.assign(pop_size, vector<double>(dim)); // 初始化current_pop
	fit.assign(pop_size, 0.0); // 初始化 fit
	current_fit.assign(pop_size, 0.0); // 初始化 current_fit

	// 產生初始解pop
	for (int i = 0; i < pop_size; ++i) {
		for (int j = 0; j < dim; ++j) {
			pop[i][j] = dist_init(gen);
			//cout << pop[i][j] << " "; // 印出初始解
		}
	}
	// 初始化donor_pop和current_pop
	donor_pop = pop;
	current_pop = pop;

	// 計算初始解的fitness
	for (int i = 0; i < pop_size; ++i) {
		//fit[i] = benchmark.Evaluate(pop[i]);
		//benchmark->Evaluate(pop[i].data(), &fit[i], 1); // void Evaluate(double* x, double* f, int mx = 1)
		benchmark.Evaluate(pop[i].data(), &fit[i], 1);
		if (i == 0 || fit[i] < best_fit) {
			// 更新最佳解
			best_fit = fit[i];
			best_sol = pop[i];
		}
	}
}

/* 產生donor解 */
void algo_DE::Mutation(
	int dim, int pop_size,
	vector<vector<double>>& pop,
	vector<vector<double>>& donor_pop,
	vector<vector<double>>& current_pop,
	vector<double>& fit,
	vector<double>& current_fit,
	double& best_fit,
	vector<double>& best_sol,
	double& lower_bound,
	double& upper_bound,
	double F
)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist_idx(0, pop_size - 1);

	for (int i = 0; i < pop_size; ++i)
	{
		int r1, r2, r3;

		do { r1 = dist_idx(gen); } while (r1 == i);
		do { r2 = dist_idx(gen); } while (r2 == i || r2 == r1);
		do { r3 = dist_idx(gen); } while (r3 == i || r3 == r2 || r3 == r1);

		for (int j = 0; j < dim; ++j)
		{
			donor_pop[i][j] = pop[r1][j]
				+ F * (pop[r2][j] - pop[r3][j]);
		}
	}
}

/* 原解和donor解交叉產生current解 */
void algo_DE::Crossover(
	int dim, int pop_size,
	vector<vector<double>>& pop,
	vector<vector<double>>& donor_pop,
	vector<vector<double>>& current_pop,
	vector<double>& fit,
	vector<double>& current_fit,
	double& best_fit,
	vector<double>& best_sol,
	double& lower_bound,
	double& upper_bound,
	double CR
)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist_idx(0, dim - 1);
	uniform_real_distribution<double> dist_real(0.0, 1.0);

	for (int i = 0; i < pop_size; ++i)
	{
		int jrand = dist_idx(gen);     /* 保證必有至少一個idx有變異 */

		for (int j = 0; j < dim; ++j)
		{
			double r = dist_real(gen);

			if (r < CR || jrand == j) /* 當隨機[0, 1]浮點數小於cross_rate或是隨機idx等於當前idx */
			{
				current_pop[i][j] = donor_pop[i][j]; /* 就變異為新解 */
			}
			else
			{
				current_pop[i][j] = pop[i][j];       /* 否則維持原解 */
			}
			// reflective boundary condition
			if (current_pop[i][j] < lower_bound) {
				current_pop[i][j] = lower_bound + (lower_bound - current_pop[i][j]);
				if (current_pop[i][j] > upper_bound) current_pop[i][j] = lower_bound;
			}
			else if (current_pop[i][j] > upper_bound) {
				current_pop[i][j] = upper_bound - (current_pop[i][j] - upper_bound);
				if (current_pop[i][j] < lower_bound) current_pop[i][j] = upper_bound;
			}

			//// jitter
			//if (fabs(donor_pop[i][j] - lower_bound) < 1e-8 || fabs(donor_pop[i][j] - upper_bound) < 1e-8) {
			//	donor_pop[i][j] += (upper_bound - lower_bound) * 0.01 * (2.0 * dist_CR(gen) - 1.0); // small random step
			//}
		}
	}
}

void algo_DE::Evaluation(
	int dim, int pop_size,
	vector<vector<double>>& pop,
	vector<vector<double>>& donor_pop,
	vector<vector<double>>& current_pop,
	vector<double>& fit,
	vector<double>& current_fit,
	double& best_fit,
	vector<double>& best_sol,
	double& lower_bound,
	double& upper_bound,
	int func_id,
	int& nfes
)
{
	Cec14TestFunc benchmark(dim, func_id);
	// 計算current解的fitness
	current_fit.clear();
	current_fit.resize(pop_size, 0.0);
	//Benchmark benchmark(func_id);
	for (int i = 0; i < pop_size; ++i)
	{
		//current_fit[i] = benchmark.Evaluate(current_pop[i]);
		//benchmark->Evaluate(current_pop[i].data(), &current_fit[i], 1); // void Evaluate(double* x, double* f, int mx = 1)
		benchmark.Evaluate(current_pop[i].data(), &current_fit[i], 1);
		nfes++;
	}
}

/* 比較新舊fitness,update population & fit，update best_sol & best_fit */
void algo_DE::Determination(
	int dim, int pop_size,
	vector<vector<double>>& pop,
	vector<vector<double>>& donor_pop,
	vector<vector<double>>& current_pop,
	vector<double>& fit,
	vector<double>& current_fit,
	double& best_fit,
	vector<double>& best_sol,
	double& lower_bound,
	double& upper_bound,
	vector<vector<double>>& eva_fit_record,
	int current_run
)
{
	for (int i = 0; i < pop_size; ++i)
	{
		if (current_fit[i] < fit[i])       /* 當current解優於原解，更新 */
		{
			fit[i] = current_fit[i];
			pop[i] = current_pop[i];

			if (current_fit[i] < best_fit) /* 當current解優於最佳解，更新 */
			{
				best_fit = current_fit[i];
				best_sol = current_pop[i];
			}
		}
		eva_fit_record[current_run].push_back(best_fit); // 記錄每次evaluation time的最佳解
	}
}
	//cout << "sF size: " << sF.size() << ", sCR size: " << sCR.size() << endl;

	// Elitism: 保留最好的解
	// If the worst fitness in the population is better than the best fitness ever found,
	// replace the corresponding individual with the best solution found.
	/*auto worst_it = std::max_element(fit.begin(), fit.end());
	if (worst_it != fit.end() && best_fit < *worst_it) {
		int idx = std::distance(fit.begin(), worst_it);
		pop[idx] = best_sol;
		fit[idx] = best_fit;
	}*/