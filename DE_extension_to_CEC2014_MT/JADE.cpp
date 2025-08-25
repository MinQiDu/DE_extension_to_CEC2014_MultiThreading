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
#include "JADE_fileoutput.h" // �]�t FileOutput �禡���ŧi

/* Constructor ��l�ƶüƤ��� */
//algo_JADE::algo_JADE()
//	: gen(rd()) /* �� rd ��l�� gen�]�O��o�u���b��l�ƦC���^ */
//{
//}
/* Destructor ����ʺA�t�m���O���� */
//algo_JADE::~algo_JADE() {
//	 if (benchmark != nullptr) {
//	 	delete benchmark;    // ����ʺA�t�m���O����
//	 	benchmark = nullptr; // �n�ߺD�A�קK������
//	 }
//}

void algo_JADE::SingleRun(int r, int _dim, int _pop_size, double _mCR, double _mF, double _c, double _p, int _func_id,
	bool _archive_flag,
	vector<double>& best_fit_record,
	vector<vector<double>>& eva_fit_record)
{
	bool t_archive_flag = _archive_flag;
	int t_func_id = _func_id;
	int t_iter_c = 0;					  /* ��e���N���� */
	int t_dim = _dim;
	int r_init_pop = 18;				  /* ��l�s��j�p��� */
	int t_pop_size = t_dim * r_init_pop;
	int t_nfes = 0;						  /* ��l�Ƶ������� */
	int t_mnfes = t_dim * 10000;			  /* �̤j�������� */
	double t_mCR = _mCR; /* adaptive parameter 1 (cross_rate) */
	double t_mF = _mF;   /* adaptive parameter 2 (= F, donor_rate) */
	vector<double> t_CR; /* �s���N�C�ӭ��骺 CR */
	vector<double> t_F;  /* �s���N�C�ӭ��骺 F */
	double t_c = _c;     /* adaptation rate */
	double t_p = _p;     /* top % pop chosen by pbest */

	vector<double> t_sCR; /* �s���N�����\��s�����骺CR */
	vector<double> t_sF;  /* �s���N�����\��s�����骺F */
	vector<double> t_delta_fit; 		  /* �s���N�����\��s�����骺 fitness �t�� */
	vector<vector<double>> t_A;           /* �s���N���Q�^�O������ѡAsize = pop_size */
	vector<vector<double>> t_pop;         /* dim*pop_size �Ӹs��� */
	vector<vector<double>> t_donor_pop;   /* �ܲ��᪺ donor �� */
	vector<vector<double>> t_current_pop; /* crossover �᪺�s�� */
	vector<double> t_fit;                 /* ��Ѫ� fitness */
	vector<double> t_current_fit;         /* �s�Ѫ� fitness */
	double t_best_fit;                    /* �����̨�fitness */
	vector<double> t_best_sol;            /* �����̨θ� */
	double t_lower_bound, t_upper_bound;  /* ���P��ƹ������W�U�� */
	int t_current_run = r;							  /* �O����e�B�榸�ơA�� Determination() �����ϥ� */
	eva_fit_record[t_current_run].reserve(t_mnfes);   /* ���C���B�檺 evaluation fitness �������t�Ŷ� */

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
		// ��smCR & mF
		ParaAdaptation(
			t_pop_size,
			t_A, t_CR, t_F, t_mCR, t_mF,
			t_c, t_p,
			t_sCR, t_sF, t_delta_fit, t_archive_flag
		); 
	}
	best_fit_record[t_current_run] = t_best_fit; // �O���Ҧ� run �B�檺�̨� fitness
	//cout << "best fitness in run " << (r + 1) << ": " << best_fit << endl;
	/*cout << "best solution found in run " << (r + 1) << ": ";
	for (const auto& val : best_sol) {
		cout << val << " ";
	}
	cout << endl;*/

}

void algo_JADE::RunALG(int _dim, int _pop_size, double _mCR, double _mF, double _c, double _p, int _func_id, bool _archive_flag)
{
	//benchmark = new Cec14TestFunc(_dim, _func_id);  // ��l�� benchmark ��ƪ���
	int run = 51; // �]�w�B�榸��
	vector<double> best_fit_record;		   // �Ω�O���C���B�檺�̨� fitness
	vector<vector<double>> eva_fit_record; // ��X�C�@ run �C��evaluation fitness ���ĵ��G��
	best_fit_record.resize(run);
	eva_fit_record.resize(run); // �Ω�O���C���B�檺 evaluation fitness

	vector<thread> threads;
	for (int r = 0; r < run; ++r)
	{
		threads.emplace_back(&algo_JADE::SingleRun, this, r, _dim, _pop_size, _mCR, _mF, _c, _p, _func_id,
			_archive_flag,
			ref(best_fit_record),
			ref(eva_fit_record));
	}
	// ���ݩҦ� thread ����
	for (auto& t : threads) {
		t.join();
	}

	// ��X���G
	double avg_best_fit = accumulate(best_fit_record.begin(), best_fit_record.end(), 0.0) / run;
	std::cout << "Avg Best fitness: " << avg_best_fit << endl;
	// ��X���ɮ�
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
	// ��l�Ƹs���
	//Benchmark benchmark(func_id);
	//benchmark.GetBounds(lower_bound, upper_bound);
	//benchmark->GetBounds(lower_bound, upper_bound); // �����ƪ����
	Cec14TestFunc benchmark(dim, func_id);
	benchmark.GetBounds(lower_bound, upper_bound);
	//cout << "Lower bound: " << lower_bound << ", Upper bound: " << upper_bound << endl;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dist_init(lower_bound, upper_bound); /* �ϥαq��Ǩ���������� */

	pop.clear();
	current_pop.clear();
	fit.clear();
	current_fit.clear();
	A.clear();   // A �s�L�h�Q�^�O������A��l����
	A.reserve(pop_size); // Reserve space for A

	pop.assign(pop_size, vector<double>(dim)); // ��l��pop
	current_pop.assign(pop_size, vector<double>(dim)); // ��l��current_pop
	fit.assign(pop_size, 0.0); // ��l�� fit
	current_fit.assign(pop_size, 0.0); // ��l�� current_fit

	CR.assign(pop_size, mCR); // ��l�� CR
	F.assign(pop_size, mF);   // ��l�� F

	// ���ͪ�l��pop
	for (int i = 0; i < pop_size; ++i) {
		for (int j = 0; j < dim; ++j) {
			pop[i][j] = dist_init(gen);
			//cout << pop[i][j] << " "; // �L�X��l��
		}
	}
	// ��l��donor_pop�Mcurrent_pop
	donor_pop = pop;
	current_pop = pop;

	// �p���l�Ѫ�fitness
	for (int i = 0; i < pop_size; ++i) {
		//fit[i] = benchmark.Evaluate(pop[i]);
		//benchmark->Evaluate(pop[i].data(), &fit[i], 1);
		benchmark.Evaluate(pop[i].data(), &fit[i], 1);

		if (i == 0 || fit[i] < best_fit) {
			// ��s�̨θ�
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
	// ����donor��
	donor_pop.clear();
	donor_pop.resize(pop_size, vector<double>(dim));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> dist_CR(mCR, 0.1); /* �`�A���� for �ͦ����骺 CR */
	cauchy_distribution<double> dist_F(mF, 0.1);   /* �_����� for �ͦ����骺 F */

	vector<vector<double>> pop_A = pop;                         /* ���ƻspop */
	pop_A.insert(pop_A.end(), A.begin(), A.end());              /* �A���WA */

	for (int i = 0; i < pop_size; ++i)
	{
		/* �ͦ����N�Ҧ����骺 CR[i] & F[i] */
		CR[i] = max(0.0, min(1.0, dist_CR(gen))); // truncate, �T�O CR[i] �b [0, 1] �d��

		do {
			F[i] = dist_F(gen);
		} while (F[i] <= 0.0 || F[i] > 1.0);       /* truncate */
		//F[i] = max(F[i], 0.1); // Clamp F to at least 0.1

		/* ��Xx_pbest, x_r1, x_r2 �Ӳ��� donor_pop */
		/* x_pbest */
		int p_num = p * pop_size;
		if (p_num < 2) p_num = 2;                                   /* �T�O�ܤ֦�2�ӭԿ���� */
		vector<pair<double, int>> fit_idx(pop_size);                /* pair = { fitness, index } */
		for (int i = 0; i < pop_size; ++i)                          /* �إ� fit_idx �H�ѱƧǦC�X���� fitness �ƦW */
		{
			fit_idx[i] = { fit[i], i };
		}
		sort(fit_idx.begin(), fit_idx.end());                       /* fitness �Ѥp->�j */

		int idx_pbest;
		do {
			uniform_int_distribution<int> dist_pbest(0, p_num - 1); /* ��Ƨ��ä��� for �ͦ�index */
			idx_pbest = fit_idx[dist_pbest(gen)].second;            /* pbest �b pop ���� index */
		} while (idx_pbest == i);                                   /* �T�Oidx_pbest���e����i���P */
		vector<double> x_pbest = pop[idx_pbest];                    /* �o�� x_pbest */

		/* x_r1 */
		uniform_int_distribution<int> dist_r1(0, pop_size - 1);
		int idx_r1;
		do {
			idx_r1 = dist_r1(gen);
		} while (idx_r1 == i || idx_r1 == idx_pbest);
		vector<double> x_r1 = pop[idx_r1];                          /* �o�� x_r1 */

		/* x_r2 */
		uniform_int_distribution<int> dist_r2(0, pop_size + A.size() - 1);
		
		int idx_r2;
		do {
			idx_r2 = dist_r2(gen);
		} while (idx_r2 == i || idx_r2 == idx_pbest || idx_r2 == idx_r1);
		vector<double> x_r2 = pop_A[idx_r2];                        /* �o�� x_r2 */

		for (int j = 0; j < dim; ++j)                               /* ��X�����骺 donor �� */
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
	// ��e�ͦ�recent��
	current_pop.clear();
	current_pop.resize(pop_size, vector<double>(dim));

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist_idx(0, dim - 1);  // ��Ƨ��ä��� for �H�����׿��
	uniform_real_distribution<double> dist_CR(0.0, 1.0); // ���ä��� for r
	for (int i = 0; i < pop_size; ++i)
	{
		int j_rand = dist_idx(gen); // �H����ܤ@�Ӻ��׶i��T�O�ܤ֦��@�Ӻ��רӦ�donor��
		for (int j = 0; j < dim; ++j)
		{
			double r = dist_CR(gen); // �ͦ��H���� r
			if (j == j_rand || r < CR[i]) // �T�O�ܤ֦��@�Ӻ��רӦ�donor��
				current_pop[i][j] = donor_pop[i][j]; // �p�G�O�H���襤�����ש�r�p��CR[i]�A�h�qdonor�Ѥ�����
			else
				current_pop[i][j] = pop[i][j]; // �_�h�q��Ѥ�����

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
	// �p��current�Ѫ�fitness
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
	// ��s�U�@�N��
	for (int i = 0; i < pop_size; ++i)
	{
		if (current_fit[i] < fit[i]) // �p�Gcurrent�Ѫ�fitness��n
		{
			if (archive_flag)
			{
				A.push_back(pop[i]); // �N�^�O������ѥ[�JA
			}
			pop[i] = current_pop[i]; // ��spop[i]��current��

			// Record fitness improvement for weighting
			delta_fit.push_back(fit[i] - current_fit[i]); // positive value

			fit[i] = current_fit[i]; // ��sfitness
			sF.push_back(F[i]);		 // �N���\��s��F�[�JsF
			sCR.push_back(CR[i]);	 // �N���\��s��CR�[�JsCR
			if (fit[i] < best_fit)   // �p�G��s�᪺fitness��n�A��sbest_fit�Mbest_sol
			{
				best_fit = fit[i];
				best_sol = pop[i];
			}
		}
		eva_fit_record[current_run].push_back(best_fit); // �O���C��evaluation time���̨θ�
	}
	//cout << "sF size: " << sF.size() << ", sCR size: " << sCR.size() << endl;

	// Elitism: �O�d�̦n����
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