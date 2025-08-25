#ifndef DE_H
#define DE_H
//#include "benchmark.h"
#include "cec14_test_func.h"	/* �]�t CEC14 ��ƪ��ŧi */ 
#include <random>				/* for random device & mt19937 */
using namespace std;

class algo_DE
{
public:
	//algo_DE();					/* constructor �ŧi */
	//~algo_DE();                 /* destructor �ŧi */
	void RunALG(int dim, int pop_size, double CR, double F, int func_id);
private:
	//random_device rd;			/*�ŧi�üƤ���*/
	//mt19937 gen;

	//Cec14TestFunc* benchmark = nullptr; /* benchmark function object pointer */

	//int func_id;				/* benchmark function id */
	//int iter_c;					/* ��e���N���� */
	//int dim;
	//int pop_size;
	//int mnfes;
	//int nfes;
	//double CR;					/* parameter 1 (cross_rate) */
	//double F;					/* parameter 2 (= F, donor_rate) */

	//vector<vector<double>> pop;         /* dim*pop_size �Ӹs��� */
	//vector<vector<double>> donor_pop;   /* �ܲ��᪺ donor �� */
	//vector<vector<double>> current_pop; /* crossover �᪺�s�� */
	//vector<double> fit;                 /* ��Ѫ� fitness */
	//vector<double> current_fit;         /* �s�Ѫ� fitness */
	//double best_fit;                    /* �����̨�fitness */
	//vector<double> best_sol;            /* �����̨θ� */

	//double lower_bound, upper_bound;    /* ���P��ƹ������W�U�� */

	////int current_run;					   /* ��e�B�� run �� */
	//vector<vector<double>> eva_fit_record; /* ��X�C�@ run �C��evaluation fitness ���ĵ��G�� */

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
		int func_id
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
		double F
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
		double CR
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
		int current_run
	);  
	void SingleRun(int r, int _dim, int _pop_size, double _CR, double _F, int _func_id,
		vector<double>& best_fit_record,
		vector<vector<double>>& eva_fit_record);
};

#endif
