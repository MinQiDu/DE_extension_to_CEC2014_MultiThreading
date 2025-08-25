#include "JADE.h"
#include <ctime>
#include <cstdlib>
#include <algorithm> /* for auto */
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>     /* for sqrt */
#include <numeric>   /* for accumulate */
#include <thread>
#include "JADE_fileoutput.h" // 包含 FileOutput 函式的宣告

/* Constructor 初始化亂數引擎 */
//algo_JADE::algo_JADE()
//	: gen(rd()) /* 用 rd 初始化 gen（記住這只能放在初始化列表中） */
//{
//}
/* Destructor 釋放動態配置的記憶體 */
//algo_JADE::~algo_JADE() {
//	 if (benchmark != nullptr) {
//	 	delete benchmark;    // 釋放動態配置的記憶體
//	 	benchmark = nullptr; // 好習慣，避免野指標
//	 }
//}

void algo_JADE::SingleRun(int r, int _dim, int _pop_size, double _mCR, double _mF, double _c, double _p, int _func_id,
	bool _archive_flag,
	vector<double>& best_fit_record,
	vector<vector<double>>& eva_fit_record)
{
	bool t_archive_flag = _archive_flag;
	int t_func_id = _func_id;
	int t_iter_c = 0;					  /* 當前迭代次數 */
	int t_dim = _dim;
	int r_init_pop = 18;				  /* 初始群體大小比例 */
	int t_pop_size = t_dim * r_init_pop;
	int t_nfes = 0;						  /* 初始化評估次數 */
	int t_mnfes = t_dim * 10000;			  /* 最大評估次數 */
	double t_mCR = _mCR; /* adaptive parameter 1 (cross_rate) */
	double t_mF = _mF;   /* adaptive parameter 2 (= F, donor_rate) */
	vector<double> t_CR; /* 存此代每個個體的 CR */
	vector<double> t_F;  /* 存此代每個個體的 F */
	double t_c = _c;     /* adaptation rate */
	double t_p = _p;     /* top % pop chosen by pbest */

	vector<double> t_sCR; /* 存此代中成功更新的個體的CR */
	vector<double> t_sF;  /* 存此代中成功更新的個體的F */
	vector<double> t_delta_fit; 		  /* 存此代中成功更新的個體的 fitness 差值 */
	vector<vector<double>> t_A;           /* 存歷代中被淘汰掉的原解，size = pop_size */
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
		t_func_id,
		t_A,
		t_CR, t_F,
		t_mCR, t_mF
	);

	while (t_nfes < t_mnfes)
	{
		Mutation(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			t_A, t_CR, t_F, t_mCR, t_mF,
			t_c, t_p
		);
		Crossover(
			t_dim, t_pop_size,
			t_pop, t_donor_pop, t_current_pop,
			t_fit, t_current_fit,
			t_best_fit, t_best_sol,
			t_lower_bound, t_upper_bound,
			t_A, t_CR, t_F, t_mCR, t_mF,
			t_c, t_p
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
			t_current_run,
			t_A, t_CR, t_F, t_mCR, t_mF, 
			t_c, t_p,
			t_sCR, t_sF, t_delta_fit, t_archive_flag
		);
		// 更新mCR & mF
		ParaAdaptation(
			t_pop_size,
			t_A, t_CR, t_F, t_mCR, t_mF,
			t_c, t_p,
			t_sCR, t_sF, t_delta_fit, t_archive_flag
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

void algo_JADE::RunALG(int _dim, int _pop_size, double _mCR, double _mF, double _c, double _p, int _func_id, bool _archive_flag)
{
	//benchmark = new Cec14TestFunc(_dim, _func_id);  // 初始化 benchmark 函數物件
	int run = 51; // 設定運行次數
	vector<double> best_fit_record;		   // 用於記錄每次運行的最佳 fitness
	vector<vector<double>> eva_fit_record; // 輸出每一 run 每次evaluation fitness 收斂結果用
	best_fit_record.resize(run);
	eva_fit_record.resize(run); // 用於記錄每次運行的 evaluation fitness

	vector<thread> threads;
	for (int r = 0; r < run; ++r)
	{
		threads.emplace_back(&algo_JADE::SingleRun, this, r, _dim, _pop_size, _mCR, _mF, _c, _p, _func_id,
			_archive_flag,
			ref(best_fit_record),
			ref(eva_fit_record));
	}
	// 等待所有 thread 結束
	for (auto& t : threads) {
		t.join();
	}

	// 輸出結果
	double avg_best_fit = accumulate(best_fit_record.begin(), best_fit_record.end(), 0.0) / run;
	std::cout << "Avg Best fitness: " << avg_best_fit << endl;
	// 輸出到檔案
	JADE_FileOutput(
		run,
		best_fit_record,
		eva_fit_record,
		_func_id,
		_dim,
		_pop_size,
		_mCR,
		_mF,
		_c,
		_p,
		_archive_flag,
		avg_best_fit
	);
}

void algo_JADE::Init(
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
	vector<vector<double>>& A,
	vector<double>& CR, vector<double>& F,double mCR, double mF
)
{
	// 初始化群體解
	//Benchmark benchmark(func_id);
	//benchmark.GetBounds(lower_bound, upper_bound);
	//benchmark->GetBounds(lower_bound, upper_bound); // 獲取函數的邊界
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
	A.clear();   // A 存過去被淘汰的個體，初始為空
	A.reserve(pop_size); // Reserve space for A

	pop.assign(pop_size, vector<double>(dim)); // 初始化pop
	current_pop.assign(pop_size, vector<double>(dim)); // 初始化current_pop
	fit.assign(pop_size, 0.0); // 初始化 fit
	current_fit.assign(pop_size, 0.0); // 初始化 current_fit

	CR.assign(pop_size, mCR); // 初始化 CR
	F.assign(pop_size, mF);   // 初始化 F

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
		//benchmark->Evaluate(pop[i].data(), &fit[i], 1);
		benchmark.Evaluate(pop[i].data(), &fit[i], 1);

		if (i == 0 || fit[i] < best_fit) {
			// 更新最佳解
			best_fit = fit[i];
			best_sol = pop[i];
		}
	}
}

void algo_JADE::Mutation(
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
	vector<vector<double>>& A,vector<double>& CR, vector<double>& F, double mCR, double mF,
	double c, double p
)
{
	// 產生donor解
	donor_pop.clear();
	donor_pop.resize(pop_size, vector<double>(dim));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> dist_CR(mCR, 0.1); /* 常態分布 for 生成個體的 CR */
	cauchy_distribution<double> dist_F(mF, 0.1);   /* 柯西分布 for 生成個體的 F */

	vector<vector<double>> pop_A = pop;                         /* 先複製pop */
	pop_A.insert(pop_A.end(), A.begin(), A.end());              /* 再接上A */

	for (int i = 0; i < pop_size; ++i)
	{
		/* 生成此代所有個體的 CR[i] & F[i] */
		CR[i] = max(0.0, min(1.0, dist_CR(gen))); // truncate, 確保 CR[i] 在 [0, 1] 範圍內

		do {
			F[i] = dist_F(gen);
		} while (F[i] <= 0.0 || F[i] > 1.0);       /* truncate */
		//F[i] = max(F[i], 0.1); // Clamp F to at least 0.1

		/* 選出x_pbest, x_r1, x_r2 來產生 donor_pop */
		/* x_pbest */
		int p_num = p * pop_size;
		if (p_num < 2) p_num = 2;                                   /* 確保至少有2個候選個體 */
		vector<pair<double, int>> fit_idx(pop_size);                /* pair = { fitness, index } */
		for (int i = 0; i < pop_size; ++i)                          /* 建立 fit_idx 以供排序列出個體 fitness 排名 */
		{
			fit_idx[i] = { fit[i], i };
		}
		sort(fit_idx.begin(), fit_idx.end());                       /* fitness 由小->大 */

		int idx_pbest;
		do {
			uniform_int_distribution<int> dist_pbest(0, p_num - 1); /* 整數均勻分布 for 生成index */
			idx_pbest = fit_idx[dist_pbest(gen)].second;            /* pbest 在 pop 中的 index */
		} while (idx_pbest == i);                                   /* 確保idx_pbest跟當前個體i不同 */
		vector<double> x_pbest = pop[idx_pbest];                    /* 得到 x_pbest */

		/* x_r1 */
		uniform_int_distribution<int> dist_r1(0, pop_size - 1);
		int idx_r1;
		do {
			idx_r1 = dist_r1(gen);
		} while (idx_r1 == i || idx_r1 == idx_pbest);
		vector<double> x_r1 = pop[idx_r1];                          /* 得到 x_r1 */

		/* x_r2 */
		uniform_int_distribution<int> dist_r2(0, pop_size + A.size() - 1);
		
		int idx_r2;
		do {
			idx_r2 = dist_r2(gen);
		} while (idx_r2 == i || idx_r2 == idx_pbest || idx_r2 == idx_r1);
		vector<double> x_r2 = pop_A[idx_r2];                        /* 得到 x_r2 */

		for (int j = 0; j < dim; ++j)                               /* 算出此個體的 donor 解 */
		{
			donor_pop[i][j] = pop[i][j] + F[i] * (x_pbest[j] - pop[i][j]) + F[i] * (x_r1[j] - x_r2[j]);

			// reflective boundary condition
			if (donor_pop[i][j] < lower_bound) {
				donor_pop[i][j] = lower_bound + (lower_bound - donor_pop[i][j]);
				if (donor_pop[i][j] > upper_bound) donor_pop[i][j] = lower_bound;
			}
			if (donor_pop[i][j] > upper_bound) {
				donor_pop[i][j] = upper_bound - (donor_pop[i][j] - upper_bound);
				if (donor_pop[i][j] < lower_bound) donor_pop[i][j] = upper_bound;
			}
		}
	}
}

void algo_JADE::Crossover(
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
	vector<vector<double>>& A, vector<double>& CR, vector<double>& F, double mCR, double mF,
	double c, double p
)
{
	// 交叉生成recent解
	current_pop.clear();
	current_pop.resize(pop_size, vector<double>(dim));

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist_idx(0, dim - 1);  // 整數均勻分布 for 隨機維度選擇
	uniform_real_distribution<double> dist_CR(0.0, 1.0); // 均勻分布 for r
	for (int i = 0; i < pop_size; ++i)
	{
		int j_rand = dist_idx(gen); // 隨機選擇一個維度進行確保至少有一個維度來自donor解
		for (int j = 0; j < dim; ++j)
		{
			double r = dist_CR(gen); // 生成隨機數 r
			if (j == j_rand || r < CR[i]) // 確保至少有一個維度來自donor解
				current_pop[i][j] = donor_pop[i][j]; // 如果是隨機選中的維度或r小於CR[i]，則從donor解中取值
			else
				current_pop[i][j] = pop[i][j]; // 否則從原解中取值

			// reflective boundary condition
			if (current_pop[i][j] < lower_bound) {
				current_pop[i][j] = lower_bound + (lower_bound - current_pop[i][j]);
				if (current_pop[i][j] > upper_bound) current_pop[i][j] = lower_bound;
			}
			else if (current_pop[i][j] > upper_bound) {
				current_pop[i][j] = upper_bound - (current_pop[i][j] - upper_bound);
				if (current_pop[i][j] < lower_bound) current_pop[i][j] = upper_bound;
			}

			// jitter
			if (fabs(donor_pop[i][j] - lower_bound) < 1e-8 || fabs(donor_pop[i][j] - upper_bound) < 1e-8) {
				donor_pop[i][j] += (upper_bound - lower_bound) * 0.01 * (2.0 * dist_CR(gen) - 1.0); // small random step
			}
		}
	}
}

void algo_JADE::Evaluation(
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
		//benchmark->Evaluate(current_pop[i].data(), &current_fit[i], 1);
		benchmark.Evaluate(current_pop[i].data(), &current_fit[i], 1);
		nfes++;
	}
}

void algo_JADE::Determination(
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
	int current_run,
	vector<vector<double>>& A, vector<double>& CR, vector<double>& F, double mCR, double mF,
	double c, double p,
	vector<double>& sCR, vector<double>& sF, vector<double>& delta_fit, bool archive_flag
)
{
    sF.reserve(pop_size);
    sCR.reserve(pop_size);
	// 更新下一代解
	for (int i = 0; i < pop_size; ++i)
	{
		if (current_fit[i] < fit[i]) // 如果current解的fitness更好
		{
			if (archive_flag)
			{
				A.push_back(pop[i]); // 將淘汰掉的原解加入A
			}
			pop[i] = current_pop[i]; // 更新pop[i]為current解

			// Record fitness improvement for weighting
			delta_fit.push_back(fit[i] - current_fit[i]); // positive value

			fit[i] = current_fit[i]; // 更新fitness
			sF.push_back(F[i]);		 // 將成功更新的F加入sF
			sCR.push_back(CR[i]);	 // 將成功更新的CR加入sCR
			if (fit[i] < best_fit)   // 如果更新後的fitness更好，更新best_fit和best_sol
			{
				best_fit = fit[i];
				best_sol = pop[i];
			}
		}
		eva_fit_record[current_run].push_back(best_fit); // 記錄每次evaluation time的最佳解
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
}

void algo_JADE::ParaAdaptation(
	int pop_size,
	vector<vector<double>>& A, vector<double>& CR, vector<double>& F, double mCR, double mF,
	double c, double p,
	vector<double>& sCR, vector<double>& sF, vector<double>& delta_fit, bool archive_flag
)
{
	random_device rd;
	mt19937 gen(rd());

    // Maintain size of A <= pop_size
    while (A.size() > pop_size)
    {
        uniform_int_distribution<int> dist_idx(0, A.size() - 1);
        int idx = dist_idx(gen);
        A.erase(A.begin() + idx);
    }

    // mCR adaptation (arithmetic mean)
    double mean_sCR = mCR;
    if (!sCR.empty()) {
        double sum_sCR = 0.0;
        for (double cr : sCR) sum_sCR += cr;
        mean_sCR = sum_sCR / sCR.size();
    }
    mCR = (1 - c) * mCR + c * mean_sCR;

    // mF adaptation (weighted Lehmer mean)
    double mean_sF = mF;
    if (!sF.empty() && !delta_fit.empty()) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < sF.size(); ++i) {
            if (sF[i] > 0.0) { // filter out non-positive F
                numerator += delta_fit[i] * sF[i] * sF[i];
                denominator += delta_fit[i] * sF[i];
            }
        }
        if (denominator > 1e-12) // avoid division by zero
            mean_sF = numerator / denominator;
    }
    mF = (1 - c) * mF + c * mean_sF;

    sCR.clear();
    sF.clear();
    delta_fit.clear();
	// Update ParaAdaptation strategy:
	// Weighted Lehmer Mean: mean_sF now uses fitness improvement (delta_fit) as weights, as per the JADE paper.
	// Filter F : Only positive F values are used.
	// Numerical Stability : Added a small epsilon to avoid division by zero.
	// delta_fit : You need to record the fitness improvement for each successful individual in Determination :
}