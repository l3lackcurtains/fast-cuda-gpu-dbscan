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
#include "report.h"
#include <sys/time.h>
#include <locale.h>

static char filename [64];

double calculate_a_time(struct timeval* t1, struct timeval* t2)
{
    return (1000000.0*(t2->tv_sec-t1->tv_sec) + t2->tv_usec-t1->tv_usec)/1000.0;

}

void calculate_times(report_t* report)
{
    {
        struct timeval t1 = report->data_read_time_begin;
        struct timeval t2 = report->data_read_time_end;
        report->data_read_time =
            (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    }

    {
        struct timeval t1 = report->processing_time_begin;
        struct timeval t2 = report->processing_time_end;
        report->processing_time =
            (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    }

    {
        struct timeval t1 = report->total_time_begin;
        struct timeval t2 = report->total_time_end;
        report->total_time =
            (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    }

    {
        struct timeval t1 = report->kernel_time_begin;
        struct timeval t2 = report->kernel_time_end;
        report->kernel_time =
            (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    }

    {
    	printf("Init Memory %f, Memory consumed %f\n",report->init_memory,report->memory_consumed);
    	report->memory_consumed = report->memory_consumed - report->init_memory;
    }


}

void set_initMemory(report_t* report, float init_memory)
{
	report->init_memory = init_memory;
}

void set_finalMemory(report_t* report, float memory_consumed)
{
	report->memory_consumed = memory_consumed;
}

void set_num_threads(report_t* report, int num_threads)
{
    report->num_threads = num_threads;
}

void set_num_blocks(report_t* report, int grid)
{
    report->num_blocks = grid;
}

void set_epsilon(report_t* report, float epsilon)
{
    report->epsilon = (double)epsilon;
}

void set_num_points(report_t* report, int num_points)
{
    report->num_points = num_points;
}

void set_num_clusters(report_t* report, int num_clusters)
{
    report->num_clusters = num_clusters;
}

void set_minPts(report_t* report, int minPts)
{
	report->minPts = minPts;
}

void initialize_report(report_t* report)
{
    report->data_read_time      = 0.0;
    report->processing_time     = 0.0;
    report->total_time          = 0.0;
    report->kernel_time = 0.0;
    report->num_threads         = 0;
    report->epsilon = 0;
    report->minPts = 0;
    report->num_blocks=0;
    report->num_clusters=0;
    report->num_points=0;
    report->num_threads=0;
    report->init_memory = 0;
    report->memory_consumed = 0;
}

void print_report(report_t* report) {
    setlocale(LC_NUMERIC, "");
    calculate_times(report);

    printf("------PERFORMANCE REPORT------\n");
    printf("  G-DBSCAN.- \n");
    printf("  Epsilon :\t\t%f ms\n", report->epsilon);
    printf("  MinPts :\t\t%d \n", report->minPts);
    printf("  Points :\t\t%d \n", report->num_points);
    printf("  Clusters :\t\t%d \n", report->num_clusters);
    printf("  Num threads:\t\t%d\n", report->num_threads);
    printf("  Num blocks:\t\t%d\n", report->num_blocks);
    printf("  Data preproc. time:\t%f s\n", report->data_read_time/1000.0);
    printf("  Processing time:\t%f s\n", report->processing_time/1000.0);
    printf("  Kernel time:\t%f s\n", report->kernel_time/1000.0);
    printf("  Total time :\t\t%f s\n", report->total_time/1000.0);
    printf("  Memory consumed:\t\t%f MB\n", report->memory_consumed);
    printf("-------------------------------------------\n");

}

void get_filename(){
	struct tm *timenow;

	time_t now = time(NULL);
	timenow = gmtime(&now);

	strftime(filename, sizeof(filename), "Results_%Y-%m-%d_%H:%M:%S.csv", timenow);
}

void init_csv(){
	get_filename();
	FILE *fp;
	fp=fopen(filename,"w+");
	fprintf(fp,"Epsilon,MinPts,Points,Clusters,Num threads,Num blocks,Data preproc. time,Processing time,Kernel time,Total time, Memory(MB)\n");
	fclose(fp);
}

void save_to_csv(report_t* report){
    FILE *fp;
    fp=fopen(filename,"a");
    fprintf(fp,"%f,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n",
    		 report->epsilon,
    		 report->minPts,
    		 report->num_points,
    		 report->num_clusters,
    		 report->num_threads,
    		 report->num_blocks,
    		 report->data_read_time,
    		 report->processing_time,
    		 report->kernel_time,
    		 report->total_time,
    		 report->memory_consumed
    );
    fclose(fp);
}

void add_csv_separation(){
	FILE *fp;
	fp=fopen(filename,"a");
	fprintf(fp,"\n");
	fclose(fp);
}

void print_report_array(report_t reports[], int length_reports )
{
    int i = 0;
    for(i = 0; i < length_reports; i++) {
        print_report(&(reports[i]));
    }
}


//void average_previous_reports(report_t report[], int start_index, int end_index)
//{
//
//    for(int i = start_index + 1; i < end_index; i++) {
//        /* report[index-num_previous+1].data_read_time  += */
//        /*     report[index-num_previous+1+i].data_read_time; */
//        report[start_index].processing_time     += report[i].processing_time;
//        report[start_index].total_time          += report[i].total_time;
//        report[start_index].filter_time         += report[i].filter_time;
//        report[start_index].sort_pruning_time   += report[i].sort_pruning_time;
//        report[start_index].check_complete_time += report[i].check_complete_time;
//        report[start_index].choose_epsilon_time += report[i].choose_epsilon_time;
//        report[start_index].refine_time         += report[i].refine_time;
//        report[start_index].num_restarts        += report[i].num_restarts;
//        report[start_index].num_threads         += report[i].num_threads;
//        report[start_index].num_points          += report[i].num_points;
//        report[start_index].num_candidates      += report[i].num_candidates;
//        report[start_index].reduction_rate      += report[i].reduction_rate;
//    }
//
//    //    report[index-num_previous+1].data_read_time  /= num_previous;
//    const int num_elems = end_index - start_index;
//    report[start_index].processing_time     /= num_elems;
//    report[start_index].total_time          /= num_elems;
//    report[start_index].filter_time         /= num_elems;
//    report[start_index].refine_time         /= num_elems;
//    report[start_index].check_complete_time /= num_elems;
//    report[start_index].choose_epsilon_time /= num_elems;
//    report[start_index].sort_pruning_time   /= num_elems;
//    report[start_index].num_restarts        /= num_elems;
//    report[start_index].num_threads         /= num_elems;
//    report[start_index].num_points          /= num_elems;
//    report[start_index].num_candidates      /= num_elems;
//    report[start_index].reduction_rate      /= num_elems;
//
//}
//
//void average_all_reports_print(report_t report[], int num_reports)
//{
//    for(int i = 1; i < num_reports; i++) {
//        report[0].data_read_time      += report[i].data_read_time;
//        report[0].processing_time     += report[i].processing_time;
//        report[0].total_time          += report[i].total_time;
//        report[0].filter_time         += report[i].filter_time;
//        report[0].sort_pruning_time   += report[i].sort_pruning_time;
//        report[0].check_complete_time += report[i].check_complete_time;
//        report[0].choose_epsilon_time += report[i].choose_epsilon_time;
//        report[0].refine_time         += report[i].refine_time;
//        report[0].num_restarts        += report[i].num_restarts;
//        report[0].num_threads         += report[i].num_threads;
//        report[0].num_points          += report[i].num_points;
//        report[0].num_candidates      += report[i].num_candidates;
//        report[0].reduction_rate      += report[i].reduction_rate;
//    }
//
//    report[0].data_read_time      /= num_reports;
//    report[0].processing_time     /= num_reports;
//    report[0].total_time          /= num_reports;
//    report[0].filter_time         /= num_reports;
//    report[0].choose_epsilon_time /= num_reports;
//    report[0].sort_pruning_time   /= num_reports;
//    report[0].check_complete_time /= num_reports;
//    report[0].refine_time         /= num_reports;
//    report[0].num_restarts        /= num_reports;
//    report[0].num_threads         /= num_reports;
//    report[0].num_points          /= num_reports;
//    report[0].num_candidates      /= num_reports;
//    report[0].reduction_rate      /= num_reports;
//
//    printf("Printing the Special report\n");
//    print_report(&report[0]);
//}
