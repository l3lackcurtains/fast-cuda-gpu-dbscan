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
#ifndef _READ_DATASET_H
#define _READ_DATASET_H

typedef struct {
    float top_left_corner_x;
    float top_left_corner_y;
    float bottom_right_corner_x;
    float bottom_right_corner_y;
    int side_num_cells;
    float cell_length_x;// The number of grid cells in 1 dimension
    float cell_length_y;// The number of grid cells in 1 dimension
} grid_t;

typedef struct {
    //int *TRI;
    //int TRI_length;
    int *TKI;
    int TKI_length;
    int *PTI;
    int PTI_length;
    float *x;
    float *y;
    float *t;
    int num_points;
    int* track_to_traj;
    int num_tracks;
    grid_t grid;
} traj_data_t;

typedef struct Time
{
    unsigned int y : 6;
    unsigned int m : 4;
    unsigned int d : 5;
    unsigned int hh : 5;
    unsigned int mm : 6;
    unsigned int ss : 6;
}Time;

typedef struct Point
{
    int id;
    int uid;
    double lon, lat;
    Time t;
}Point;

int preprocess_data(char* filename, traj_data_t** data, int max_pts_limit, int repeat);
void move_data_to_gpu(traj_data_t** h_data, traj_data_t** d_data);


#endif
