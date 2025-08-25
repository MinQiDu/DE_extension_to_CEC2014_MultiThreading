#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

void JADE_FileOutput(
	const int run,
	const std::vector<double>& best_fit_record,
	const std::vector<std::vector<double>>& eva_fit_record,
	int func_id, int dim, int pop_size,
	double _mCR, double _mF, double c, double p,
	bool archive_flag, double avg_best_fit
) {
	// ===== ���� .txt ���{���X =====
	// ��X�C run �B�檺�̨� fitness �� JADE_fitness_func_dim_archive_.txt
	ofstream output_file("JADE_fitness_func" + to_string(func_id) + "_dim" + to_string(dim) + "_archive_" + (archive_flag ? "true" : "false") + ".txt", ios::out);
	if (!output_file) {
		cerr << "Error opening output file." << endl;
		return;
	}
	output_file << "JADE | func_id = " << func_id << " | dim = " << dim << " | pop_size = " << pop_size
		<< " | mCR = " << _mCR << " | mF = " << _mF << " | c = " << c << " | p = " << p
		<< " | w/o archive list = " << (archive_flag ? "true" : "false") << endl;
	output_file << "Avg Best fitness: " << avg_best_fit << endl;
	for (int i = 0; i < run; ++i) {
		output_file << " | Best fitness run " << (i + 1) << ": " << best_fit_record[i] << endl;
	}
	output_file.close();

	// ��X��X���G�� JADE_integrated_fitness.txt
	ofstream integrated_output("JADE_integrated_fitness.txt", ios::app);
	if (!integrated_output) {
		cerr << "Error opening integrated output file." << endl;
		return;
	}
	integrated_output << "JADE | func_id = " << func_id << " | dim = " << dim << " | pop_size = " << pop_size
		<< " | mCR = " << _mCR << " | mF = " << _mF << " | c = " << c << " | p = " << p
		<< " | w/o archive list = " << (archive_flag ? "true" : "false") << endl;
	integrated_output << "Avg Best fitness: " << avg_best_fit << endl;
	integrated_output.close();

	///* �X��(�C�@ run ���}) */
	//// ��X�C run �B�檺 fitness �� JADE_fitness_cvg.txt
	//ofstream cvg_file("JADE_fitness_cvg" + to_string(func_id) + "_iter" + to_string(iter) + "_dim" + to_string(dim) + "_archive_" + (archive_flag ? "true" : "false") + ".txt");
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
	//ofstream plot_file("plot_JADE_func" + to_string(func_id) + "_iter" + to_string(iter) + "_dim" + to_string(dim) + "_archive_" + (archive_flag ? "true" : "false") + ".plt");
	//if (!plot_file) {
	//	cerr << "Error opening plot file." << endl;
	//	return;
	//}
	//plot_file << "set terminal pngcairo size 1200,600 enhanced font 'Verdana,10'\n";
	//plot_file << "set output 'JADE_cvg_plot" << "_func" << func_id << "_iter" << iter << "_dim" << dim << "_archive_" << (archive_flag ? "true" : "false") << ".png'\n";
	//plot_file << "set title 'JADE Convergence Plot" << " (f" << func_id << ", iter" << iter << ", dim" << dim << ", archive: " << (archive_flag ? "true" : "false") << ")'\n";
	//plot_file << "set xlabel 'Iteration'\n";
	//plot_file << "set ylabel 'Best Fitness'\n";
	//plot_file << "set grid\n";
	//plot_file << "set label 'Avg best fitness: " << avg_best_fit << "' at graph 0.02, graph 0.95 font ',10' tc rgb 'black'\n";
	//plot_file << "set lmargin 10\n";
	//plot_file << "set tmargin 5\n";
	//plot_file << "plot ";
	//for (int i = 2; i <= run + 1; ++i) {
	//	plot_file << "'JADE_fitness_cvg" << to_string(func_id) << "_iter" << to_string(iter) << "_dim" << to_string(dim) << "_archive_" << (archive_flag ? "true" : "false") << ".txt' using :" << i
	//		<< " with lines title 'Run " << (i - 1) << "'";
	//	if (i != run + 1) plot_file << ", ";
	//}
	//plot_file << "\n";
	//plot_file.close();

	/* �X��(�Ҧ� run ����) */
	ofstream avg_cvg_file("JADE_fitness_avg_cvg" + to_string(func_id) + "_dim" + to_string(dim) + "_archive_" + (archive_flag ? "true" : "false") + ".txt");
	if (!avg_cvg_file) {
		cerr << "Error opening convergence plot file." << endl;
		return;
	}

	// �p��Ҧ� run �C�@�� iteration ��������
	int max_eva = eva_fit_record[0].size();
	for (int t = 0; t < max_eva; ++t) {
		double sum = 0.0;
		for (int r = 0; r < run; ++r) {
			sum += eva_fit_record[r][t];
		}
		avg_cvg_file << sum / run << "\n";
	}
	avg_cvg_file.close();

	// ���� plot_avg_JADE.plt �e�X�������Ĺ�
	ofstream avg_plot_file("plot_avg_JADE_func" + to_string(func_id) + "_dim" + to_string(dim) + "_archive_" + (archive_flag ? "true" : "false") + ".plt");
	if (!avg_plot_file) {
		cerr << "Error opening average plot file." << endl;
		return;
	}
	avg_plot_file << "set terminal pngcairo size 1200,600 enhanced font 'Verdana,10'\n";
	avg_plot_file << "set output 'JADE_avg_cvg_plot" << "_func" << func_id << "_dim" << dim << "_archive_" << (archive_flag ? "true" : "false") << ".png'\n";
	avg_plot_file << "set title 'JADE Average Convergence Plot" << " (f" << func_id << ", dim" << dim << ", archive: " << (archive_flag ? "true" : "false") << ")'\n";
	avg_plot_file << "set xlabel 'Evaluation times'\n";
	avg_plot_file << "set ylabel 'Average Best Fitness'\n";
	avg_plot_file << "set grid\n";
	avg_plot_file << "set label 'Avg best fitness: " << avg_best_fit << "' at graph 0.02, graph 0.95 font ',10' tc rgb 'black'\n";
	avg_plot_file << "set lmargin 10\n";
	avg_plot_file << "set tmargin 5\n";
	avg_plot_file << "set logscale y\n";
	avg_plot_file << "set xrange [0:" << max_eva << "]\n";
	avg_plot_file << "plot 'JADE_fitness_avg_cvg" << to_string(func_id) << "_dim" << to_string(dim) << "_archive_" << (archive_flag ? "true" : "false") << ".txt' with lines title 'Average Best Fitness'\n";
	avg_plot_file.close();
}
