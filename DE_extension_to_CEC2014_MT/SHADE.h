#ifndef SHADE_H
#define SHADE_H
//#include "benchmark.h"
#include "cec14_test_func.h" /* �]�t CEC14 ��ƪ��ŧi */ 
#include <random>			 /* for random device & mt19937 */
using namespace std;

class algo_SHADE
{
public:
	//algo_SHADE();       /* constructor �ŧi */
	//~algo_SHADE();      /* destructor �ŧi */
	void RunALG(int dim, int pop_size, double mCR, double mF, double c, double p, int func_id, bool archive_flag);
private:
	//random_device rd;  /*�ŧi�üƤ���*/
	//mt19937 gen;

	//Cec14TestFunc* benchmark = nullptr; /* benchmark function object pointer */

	//bool archive_flag; /* �O�_�ϥ� archive list A */
	//int func_id; /* benchmark function id */
	//int iter_c;  /* ��e���N���� */
	//int dim;
	//int pop_size;
	//int mnfes;
	//int nfes;
	////double mCR;        /* adaptive parameter 1 (cross_rate) */
	////double mF;         /* adaptive parameter 2 (= F, donor_rate) */
	//vector<double> CR; /* �s���N�C�ӭ��骺 CR */
	//vector<double> F;  /* �s���N�C�ӭ��骺 F */
	////double c;          /* adaptation rate */
	//int H;			   /* MCR & MF �j�p */
	//double p;          /* top % pop chosen by pbest */
	//vector<double> MCR; /* �s��V�@�N�����\mCR */
	//vector<double> MF;  /* �s��V�@�N�����\mF */
	//int k; /* �n�Q�۾A����s MCR & MF �� index */

	//vector<double> sCR;                 /* �s���N�����\��s�����骺CR */
	//vector<double> sF;                  /* �s���N�����\��s�����骺F */
	//vector<double> delta_fit; 		    /* �s���N�����\��s�����骺 fitness �t�� */
	//vector<vector<double>> A;           /* �s���N���Q�^�O������ѡAsize = pop_size */
	//vector<vector<double>> pop;         /* dim*pop_size �Ӹs��� */
	//vector<vector<double>> donor_pop;   /* �ܲ��᪺ donor �� */
	//vector<vector<double>> current_pop; /* crossover �᪺�s�� */
	//vector<double> fit;                 /* ��Ѫ� fitness */
	//vector<double> current_fit;         /* �s�Ѫ� fitness */
	//double best_fit;                    /* �����̨�fitness */
	//vector<double> best_sol;            /* �����̨θ� */

	//double lower_bound, upper_bound;    /* ���P��ƹ������W�U�� */

	//vector<vector<double>> eva_fit_record;  /* ��X�C�@ evaluation fitness ���ĵ��G�� */
	//int current_run;						/* ��e�B�� run �� */

	/* ���ͪ�ldim*pop_size�Ӹs��� */
	void Init(
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
		vector<double>& CR, vector<double>& F,
		vector<double>& MCR, vector<double>& MF, int H, int& k
	);
	/* ����donor�� */
	void Mutation(
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
		vector<vector<double>>& A, vector<double>& CR, vector<double>& F,
		double p,
		vector<double>& MCR, vector<double>& MF, int H, int& k
	);
	/* ��� & donor�� ��e�ͦ�recent�� */
	void Crossover(
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
		vector<vector<double>>& A, vector<double>& CR, vector<double>& F,
		double H, double p
	);
	/* �p��recent�Ѫ� fitness */
	void Evaluation(
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
	);
	/* update��(��� or recent��)�i�����U�@�N�ѡAupdate best_fit */
	void Determination(
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
		vector<vector<double>>& A, vector<double>& CR, vector<double>& F,
		int H, double p,
		vector<double>& sCR, vector<double>& sF, vector<double>& delta_fit, bool archive_flag
	);
	/* ��s mCR & mF */
	void ParaAdaptation(
		int pop_size,
		vector<vector<double>>& A, vector<double>& CR, vector<double>& F,
		double p,
		vector<double>& sCR, vector<double>& sF, vector<double>& delta_fit, bool archive_flag,
		vector<double>& MCR, vector<double>& MF, int H, int& k
	);
	void SingleRun(int r, int _dim, int _pop_size, double _H, double _p, int _func_id,
		bool _archive_flag,
		vector<double>& best_fit_record,
		vector<vector<double>>& eva_fit_record);
};

#endif
