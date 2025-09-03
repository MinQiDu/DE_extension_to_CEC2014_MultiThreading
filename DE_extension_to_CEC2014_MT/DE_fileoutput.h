#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

void DE_FileOutput(
	const int run,
	const vector<double>& best_fit_record,
	const vector<vector<double>>& eva_fit_record,
	int func_id, int dim, int pop_size,
	double _CR, double _F, double avg_best_fit
) {
	// ===== ���� .txt ���{���X =====
	// ��X�C run �B�檺�̨� fitness �� DE_fitness_func_dim.txt
	ofstream output_file("DE_fitness_func" + to_string(func_id) + "_dim" + to_string(dim) + ".txt", ios::out);
	if (!output_file) {
		cerr << "Error opening output file." << endl;
		return;
	}
	output_file << "DE | func_id = " << func_id << " | dim = " << dim << " | pop_size = " << pop_size
		<< " | CR = " << _CR << " | F = " << _F << endl;
	output_file << "Avg Best fitness: " << avg_best_fit << endl;
	for (int i = 0; i < run; ++i) {
		output_file << " | Best fitness run " << (i + 1) << ": " << best_fit_record[i] << endl;
	}
	output_file.close();

	// ��X��X���G�� DE_integrated_fitness.txt
	ofstream integrated_output("DE_integrated_fitness.txt", ios::app);
	if (!integrated_output) {
		cerr << "Error opening integrated output file." << endl;
		return;
	}
	integrated_output << "DE | func_id = " << func_id << " | dim = " << dim << " | pop_size = " << pop_size
		<< " | CR = " << _CR << " | F = " << _F << endl;
	integrated_output << "Avg Best fitness: " << avg_best_fit << endl;
	integrated_output.close();

	///* �X��(�C�@ run ���}) */
	//// ��X�C run �B�檺 fitness �� DE_fitness_cvg.txt
	//ofstream cvg_file("DE_fitness_cvg" + to_string(func_id) + "_iter" + to_string(iter) + "_dim" + to_string(dim) + ".txt");
	//if (!cvg_file) {
	//	cerr << "Error opening fitness convergence file." << endl;
	//	return;
	//}

	//// ����m��X�G�C�@�C�O iteration�A�欰�C run
	//for (int t = 0; t < iter; ++t) {
	//	for (int r = 0; r < run; ++r) {
	//		cvg_file << iter_fit_record[r][t];
	//		if (r != run - 1) cvg_file << " ";
	//	}
	//	cvg_file << "\n";
	//}
	//cvg_file.close();

	//// ���� plot_JADE.plt �e�X���Ĺ�
	//ofstream plot_file("plot_DE_func" + to_string(func_id) + "_iter" + to_string(iter) + "_dim" + to_string(dim) + ".plt");
	//if (!plot_file) {
	//	cerr << "Error opening plot file." << endl;
	//	return;
	//}
	//plot_file << "set terminal pngcairo size 1200,600 enhanced font 'Verdana,10'\n";
	//plot_file << "set output 'DE_cvg_plot" << "_func" << func_id << "_iter" << iter << "_dim" << dim << ".png'\n";
	//plot_file << "set title 'DE Convergence Plot" << " (f" << func_id << ", iter" << iter << ", dim" << dim << ")'\n";
	//plot_file << "set xlabel 'Iteration'\n";
	//plot_file << "set ylabel 'Best Fitness'\n";
	//plot_file << "set grid\n";
	//plot_file << "set label 'Avg best fitness: " << avg_best_fit << "' at graph 0.02, graph 0.95 font ',10' tc rgb 'black'\n";
	//plot_file << "set lmargin 10\n";
	//plot_file << "set tmargin 5\n";
	//plot_file << "plot ";
	//for (int i = 2; i <= run + 1; ++i) {
	//	plot_file << "'DE_fitness_cvg" << to_string(func_id) << "_iter" << to_string(iter) << "_dim" << to_string(dim) << ".txt' using :" << i
	//		<< " with lines title 'Run " << (i - 1) << "'";
	//	if (i != run + 1) plot_file << ", ";
	//}
	//plot_file << "\n";
	//plot_file.close();

	/* �X��(�Ҧ� run ����) */
	ofstream avg_cvg_file("DE_fitness_avg_cvg" + to_string(func_id) + "_dim" + to_string(dim) + ".txt");
	if (!avg_cvg_file) {
		cerr << "Error opening average convergence file." << endl;
		return;
	}
	
	// �p��Ҧ� run �C�@�� evaluation ��������
	int max_eva = eva_fit_record[0].size();
	for (int t = 0; t < max_eva; ++t) {
		double sum = 0.0;
		for (int r = 0; r < run; ++r) {
			sum += eva_fit_record[r][t];
		}
		double avg = sum / run;
		if (avg < 1e-99) avg = 1e-60; // 0 ��Ƭ� 1e-60 �קK logscale �ɥX��
		avg_cvg_file << avg << "\n";
	}
	avg_cvg_file.close();

	// ���� plot_avg_DE.plt �e�X�������Ĺ�
	ofstream avg_plot_file("plot_avg_DE_func" + to_string(func_id) + "_dim" + to_string(dim) + ".plt");
	if (!avg_plot_file) {
		cerr << "Error opening average plot file." << endl;
		return;
	}
	avg_plot_file << "set terminal pngcairo size 1200,600 enhanced font 'Verdana,10'\n";
	avg_plot_file << "set output 'DE_avg_cvg_plot" << "_func" << func_id << "_dim" << dim << ".png'\n";
	avg_plot_file << "set title 'DE Average Convergence Plot" << " (f" << func_id << ", dim" << dim << ")'\n";
	avg_plot_file << "set xlabel 'Evaluation times'\n";
	avg_plot_file << "set ylabel 'Average Best Fitness'\n";
	avg_plot_file << "set grid\n";
	avg_plot_file << "set label 'Avg best fitness: " << avg_best_fit << "' at graph 0.02, graph 0.95 font ',10' tc rgb 'black'\n";
	avg_plot_file << "set lmargin 10\n";
	avg_plot_file << "set tmargin 5\n";
	avg_plot_file << "set logscale y\n";
	avg_plot_file << "set xrange [0:" << max_eva << "]\n";
	avg_plot_file << "plot 'DE_fitness_avg_cvg" << to_string(func_id) << "_dim" << to_string(dim) << ".txt' with lines title 'Average Best Fitness'\n";
	avg_plot_file.close();
}
