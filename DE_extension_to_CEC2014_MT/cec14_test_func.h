// cec14_test_func.h
#ifndef CEC14_TEST_FUNC_H
#define CEC14_TEST_FUNC_H
#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
using namespace std;

#define INF 1.0e99
#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

class Cec14TestFunc {
public:
    Cec14TestFunc(int nx, int func_id);   // �o�̸��ɡϰt�m�w��
    ~Cec14TestFunc();

    // �������ª� cec14_test_func(x,f,nx=�T�w,mx,func_id=�T�w)
    void Evaluate(double* x, double* f, int mx = 1) const;
    void GetBounds(double& lower, double& upper) const;

private:
    // �T�w�Ѽ�
    int nx_ = 0, func_id_ = 0;
    int cf_num_ = 10; // ���ݭn�b�غc�l�� func_id_ �վ�

    // �w�ĻP��Ʊq*��}���Ч令vector�]�C�ӹ�Ҥ@���^
    vector<double> OShift_, M_;    // �ؤo�̨�ƦӲ�
    vector<int>    SS_;
    mutable vector<double> y_, z_, x_bound_; // �u�@�Ȧs�]mutable �� Evaluate �i��^

    // �u��]��� free functions �h�i�ӡ^
    void sphere_func(double*, double*, int, const double*, const double*, int, int) const;        /* Sphere */
    void ellips_func(double*, double*, int, const double*, const double*, int, int) const;        /* Ellipsoidal */
    void bent_cigar_func(double*, double*, int, const double*, const double*, int, int) const;    /* Discus */
    void discus_func(double*, double*, int, const double*, const double*, int, int) const;        /* Bent_Cigar */
    void dif_powers_func(double*, double*, int, const double*, const double*, int, int) const;    /* Different Powers */
    void rosenbrock_func(double*, double*, int, const double*, const double*, int, int) const;    /* Rosenbrock's */
    void schaffer_F7_func(double*, double*, int, const double*, const double*, int, int) const;   /* Schwefel's F7 */
    void ackley_func(double*, double*, int, const double*, const double*, int, int) const;        /* Ackley's */
    void rastrigin_func(double*, double*, int, const double*, const double*, int, int) const;     /* Rastrigin's  */
    void weierstrass_func(double*, double*, int, const double*, const double*, int, int) const;   /* Weierstrass's  */
    void griewank_func(double*, double*, int, const double*, const double*, int, int) const;      /* Griewank's  */
    void schwefel_func(double*, double*, int, const double*, const double*, int, int) const;      /* Schwefel's */
    void katsuura_func(double*, double*, int, const double*, const double*, int, int) const;      /* Katsuura */
    void bi_rastrigin_func(double*, double*, int, const double*, const double*, int, int) const;  /* Lunacek Bi_rastrigin */
    void grie_rosen_func(double*, double*, int, const double*, const double*, int, int) const;    /* Griewank-Rosenbrock  */
    void escaffer6_func(double*, double*, int, const double*, const double*, int, int) const;     /* Expanded Scaffer??s F6  */
    void step_rastrigin_func(double*, double*, int, const double*, const double*, int, int) const;/* Noncontinuous Rastrigin's  */
    void happycat_func(double*, double*, int, const double*, const double*, int, int) const;      /* HappyCat */
    void hgbat_func(double*, double*, int, const double*, const double*, int, int) const;         /* HGBat  */

    void hf01(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 1 */
    void hf02(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 2 */
    void hf03(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 3 */
    void hf04(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 4 */
    void hf05(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 5 */
    void hf06(double*, double*, int, const double*, const double*, const int*, int, int) const;   /* Hybrid Function 6 */

    void cf01(double*, double*, int, const double*, const double*, int) const; /* Composition Function 1 */
    void cf02(double*, double*, int, const double*, const double*, int) const; /* Composition Function 2 */
    void cf03(double*, double*, int, const double*, const double*, int) const; /* Composition Function 3 */
    void cf04(double*, double*, int, const double*, const double*, int) const; /* Composition Function 4 */
    void cf05(double*, double*, int, const double*, const double*, int) const; /* Composition Function 5 */
    void cf06(double*, double*, int, const double*, const double*, int) const; /* Composition Function 6 */
    void cf07(double*, double*, int, const double*, const double*, const int*, int) const; /* Composition Function 7 */
    void cf08(double*, double*, int, const double*, const double*, const int*, int) const; /* Composition Function 8 */

    void shiftfunc(double*, double*, int, const double*) const;
    void rotatefunc(double*, double*, int, const double*) const;
    void sr_func(double*, double*, int, const double*, const double*, double, int, int) const; /* shift and rotate */
    void asyfunc(double*, double* x, int, double) const;
    void oszfunc(double*, double*, int) const;
    void cf_cal(double*, double*, int, const double*, const double*, const double*, double*, int) const;

    // ������l��
    void load_all_files();  // �� func_id_ Ū M/OShift/SS�]�`�N�� "%lf" Ū double�^
    void alloc_buffers();   // �t y_/z_/x_bound_
};
#endif