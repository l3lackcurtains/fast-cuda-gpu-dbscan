////////////////////////////////////////////////////////////////////////////
//	  Copyright (C) 2018 Eleazar Leal, Hamza Mustafa.  All rights reserved.
//
//    This file is part of FastTopK.
//
//    FastTopK is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    FastTopK is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with FastTopK.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////////
#ifndef _REPORT_H
#define _REPORT_H

#include <sys/time.h>
#include <time.h>
#include <stdio.h>

typedef struct {
    // The total time ms taken while reading the data
    struct timeval data_read_time_begin;
    struct timeval data_read_time_end;
    double data_read_time;
    // The time ms taken by the processing time
    struct timeval processing_time_begin;
    struct timeval processing_time_end;
    double processing_time;
    // The total time ms taken
    struct timeval total_time_begin;
    struct timeval total_time_end;
    double total_time;
    // Time taken by Kernel
    struct timeval kernel_time_begin;
    struct timeval kernel_time_end;
    double kernel_time;
    // The value epsilons chosen
    double epsilon;
    // The value of minPts chosen
    int minPts;
    // The number of threads used per block
    int num_threads;
    // The number of blocks used
    int num_blocks;
    // The number of points
    int num_points;
    // The number of points
    int num_clusters;

    // Memory consumption
    float init_memory;
    float memory_consumed;
} report_t;

double calculate_a_time(struct timeval* t1, struct timeval* t2);

void calculate_times(report_t* report);
void calculate_reduction_rate(report_t* report);

void initialize_report(report_t* report);

void print_report(report_t* report);

void init_csv();

void save_to_csv(report_t* report);

void add_csv_separation();

void print_report_array(report_t reports[], int length_reports );

void set_num_threads(report_t* report, int num_threads);

void set_epsilon(report_t* report, float epsilon);

void set_num_blocks(report_t* report, int grid);

void set_num_points(report_t* report, int num_points);

void set_minPts(report_t* report, int minPts);

void set_initMemory(report_t* report, float init_memory);

void set_finalMemory(report_t* report, float memory_consumed);

void set_num_clusters(report_t* report, int num_clusters);
#endif
