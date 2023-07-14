from matplotlib.backends.backend_pdf import * #needed for savefig to pdf
import webbrowser

from logging import raiseExceptions
from re import L

import matplotlib.patches
from utils.h_find_plateaus import analyze_variances_in_rampup, get_candidates_regions_add_to_df, find_jumps_positions, match_frames_to_plateaus, match_stats_to_frames_and_plateaus
from utils.h_analysis_of_frames_and_video_properties import set_and_get_frame_of_interest, print_data_of_interest_from_cap, analyze_px_roi_surf, analyze_average_roi, analyze_px_roi_scinti, find_center_LED, find_center_of_mass_LED
from utils.make_histograms import make_hist_im_stack

import utils.h_colorcodes

import json
import pickle
import argparse

import os
import sys
import shutil #for copying files

import cv2
import numpy as np
from math import exp, nan
from more_itertools import pairwise
import scipy.ndimage
import pandas as pd
from prettytable import PrettyTable 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#import matplotlib.patches as mplpatches
#import matplotlib.ticker as tkr

from moviepy.editor import *

import warnings
from progress.bar import Bar

import pdb

################## parser #####################
parser = argparse.ArgumentParser()
parser.add_argument('experimentFile')
args = parser.parse_args()


################################################################
########## Functions for Pre-Processing ########################
##### means frames processing and writing the data frames ######
################################################################

def fix_OS_specific_paths(path):
    dir_and_base, extension = os.path.splitext(path)
    dir_name, base_filename = os.path.split(dir_and_base)
    return os.path.join(dir_name, base_filename)+extension

## Video analysis --> in utils\h_analysis_of_frames_and_video_properties.py
## Plateau analysis for reference --> utils\h_find_plateaus.py

## Some general things
def set_boundaries_roi(upperleft_x, upperleft_y, lowerright_x, lowerright_y):
    ''' use this function later to hand over values when marking them in image instead of definining manually '''
    bound = [upperleft_x, upperleft_y, lowerright_x, lowerright_y]
    return bound

# video cutting/copying/preparation
def cut_video(path):
    clip_tbc = VideoFileClip(path)
    print("File loaded: ", path)
    print("Duration of loaded video: ", str(clip_tbc.duration))
    continue_flag = input("Dou want to cut the video and write a cutted copy? (Yes/No) <- case sensitive!\n")
    if continue_flag == "Yes":
        first_cut= input("Second where to cut in the beginning: ")
        second_cut = input("Second where to cut at the end: ")
        clip_cut = clip_tbc.subclip(first_cut, second_cut)
        path_vid, path_vid_ext =  os.path.splitext(path)
        newPath = path_vid + '_cut_from'+str(first_cut)+'_to'+str(second_cut) + path_vid_ext
        clip_cut.write_videofile(newPath)
        return newPath
    else:
        return path

def copy_input_video(path):
    ''' loads the video from path with moviepy, copies it, adds a copy indicator to the name and saves it to same location
        prints FPS
        '''
    clip = VideoFileClip(path) # load the video file
    print("File loaded: ", path)
    print("Duration of loaded video: ", str(clip.duration))
    cut_flag = input("Dou want to cut the video and write a cutted copy? (Yes/No) <- case sensitive!")
        
    if cut_flag == "Yes":
        first_cut= input("Second where to cut in the beginning: ")
        second_cut = input("Second where to cut at the end: ")
        clip_cut = clip.subclip(first_cut, second_cut)
        clip = clip_cut

    namePath, extensionPath = os.path.splitext(path) # split rename cioy
    nameCopyPath = namePath+r"_copyByMoviepy"
    if cut_flag == "Yes":
        nameCopyPath = nameCopyPath + '_cut_from' +str(first_cut)+'_to'+str(second_cut)
    newPath = nameCopyPath+extensionPath
    
    clip.write_videofile(newPath)
    clip_check = VideoFileClip(newPath)
    print("clipcheck FPS: ", str(clip_check.fps))
    return newPath

#####################################################################
############# Functions for Post-Processing #########################
#####################################################################
# ---------------- Prepare the Data Frame -----------#
def add_scinti_state_to_df(df, scinti_th):
    for row in df.itertuples():
        if df.at[row.Index, "px_roi_scinti_normalized"] >= scinti_th:
            df.loc[row.Index, "scinti_on"]= True
        else:
            df.loc[row.Index, "scinti_on"] = False

def add_scinti_cycle_to_df(df, max_gap_in_beam):
    indx_scinti_on = df[df['scinti_on']].index

    cycles_scinti_on = []
    chunk = []
    for v1, v2 in pairwise(indx_scinti_on.tolist()):
        if v2 == indx_scinti_on.max():
            chunk.append(v1)
            chunk.append(v2)
            #fill voids
            last_elem = chunk[-1]
            chunk_filled = list(range(chunk[0],chunk[-1]))
            chunk_filled += [last_elem]
            cycles_scinti_on.append(chunk_filled)
            break
        if v2 - v1 <= max_gap_in_beam:
            chunk.append(v1)
        else:
            chunk.append(v1)
            #fill voids
            last_elem = chunk[-1]
            chunk_filled = list(range(chunk[0],chunk[-1]))
            chunk_filled += [last_elem]
            cycles_scinti_on.append(chunk_filled)
            chunk=[]
    
    #pdb.set_trace()
    cycle_cnt=1
    cycle_numbers = []
    for num in df.index:
        if cycle_cnt <= len(cycles_scinti_on):
            if num in cycles_scinti_on[(cycle_cnt-1)]:
                cycle_numbers.append(cycle_cnt)
                if num == cycles_scinti_on[(cycle_cnt-1)][-1]: #last element of cycle
                    cycle_cnt +=1
            else: 
                cycle_numbers.append(nan)
        else:
            cycle_numbers.append(nan)
            #print(num, cycle_numbers[-1])

    if len(cycle_numbers) != df.shape[0]:
        print("SOMETHING IS WRONG HERE!")
        pdb.set_trace()
    
    df_with_cycles = df.assign(scinti_on_cycle=cycle_numbers)
    
    return df_with_cycles

def add_breathing_cycles(df):
    #TODO extend breathing cycles to following local minimum?


    #start with 0. start a new one after scinti_on_cycle stops
    
    #get last indices of each scinti_On_period:
    last_indices_of_scinti_cycles = df.groupby('scinti_on_cycle', dropna = True).tail(1).index

    # Create a list of tuples with (group name, last index)
    list_tuples_groupnames_last_indices = list(zip(df.loc[last_indices_of_scinti_cycles, 'scinti_on_cycle'], last_indices_of_scinti_cycles))

    df['breathing_cycle'] = None
    #iterate over tuple/last indices and fill in new colum breathing_cycles
    for i, (cyclename, last_index) in enumerate(list_tuples_groupnames_last_indices):
        if i == 0:
            df.loc[:last_index, 'breathing_cycle']  = cyclename #first
        else:
            before_last_index = last_indices_of_scinti_cycles[i-1]
            df.loc[before_last_index+1:last_index, 'breathing_cycle'] = cyclename


#------------ Reference Analysis ---------------------#
def add_plateaus_to_reference(df, len_plateau, fps_ref, padding, reg, exclude):
    ''' uses utils\h_find_plateaus.py extensively'''
    len_win = len_plateau*fps_ref
    # define rampup region:
    '''
    if reg==[0,0]:
        # - plot reference and ask for seconds as in put
        print('Look at the following plot and note down, where the reference region starts and where it ends.')
        #plt.plot(df['frame_times']/1000, df['px_roi_surf'],'.', color='orange')
        #plt.xlabel('time [s]')
        plt.plot(df['px_roi_surf'], linestyle= ' ', marker='.', color='orange')
        plt.plot(df['px_roi_scinti']/df['px_roi_scinti'].max()*df['px_roi_surf'].max(),  linestyle= ' ', marker = '.', color= 'blue')
        plt.title('Define reference region')
        plt.xlabel('frames [1]')
        plt.show()
        #a = int(input("Which second does the reference region start?"))
        #b = int(input("Which second does the reference region stop?"))    
        #a= int(a*1000/8.3)
        #b= int(b*1000/8.3)
        a = int(input("At which frame does the reference region start?"))
        b = int(input("At which frame does the reference region stop?"))
    else:
        a = reg[0]
        b = reg[1]
        print('a=', a)
        print('b=', b)
        
    # define rampup region (plot the curve and )
    #TODO: check mal, ob das eigentlich so richtig ist, oder ob ich nicht lieber frames wählen sollte
    df_rampup = df[a:b]
    '''
    # analyze rampup roughly to find jump candidates
    df_rampup_mod = analyze_variances_in_rampup(df, len_win, fps_ref, exclude)
    
    #identify groups/step candidates, by finding changes in True/False in variance and write to df:
    df_rampup_mod = get_candidates_regions_add_to_df(df_rampup_mod,exclude)

    # find exact jumps poistions and values:
    find_jumps_positions(df_rampup_mod)

    # find exact jumps poistions and values for rm finds:
    match_frames_to_plateaus(df_rampup_mod)
    df_with_plateaus = match_stats_to_frames_and_plateaus(df_rampup_mod, padding=padding)

    #get statistics of plateaus
    df_plateaus_stats = df_rampup_mod.groupby('plateauWpad', dropna=True)['px_roi_surf'].describe()
    df_plateaus_stats['SEM'] = df_plateaus_stats['std']/np.sqrt(df_plateaus_stats['count'])

    return df_with_plateaus, df_plateaus_stats
"""
def analyze_reference(df, total_num_frames_ref, max_ref, frame_rate, oneSec_bias = True):
    '''
    simple prototype #TODO to find the lower level of the gating window
    #TODO: without onesec bias
    Idea: whenever scinti is on, get the minimal surf/LED value of the 120 frames before (if oneSec_bias == True). 
    why: EXTD awaits the surface to be a minimum of one second within the gating window until sending gating signal to Linac
     exclude those scinti ons after which surf/LED has reached 90% of the maximum (to avoid the falling edge of the signal)

    best practice: reference video should be full amplitude as std video

    # with Brainlab bias: minimal value of the last second -> always need the smallest value of the last second

    Args:
        df          (pd.data_frame) :   full frame analyses series from video. Here: reference video
        max_ref     (int)           :   expected maximum of reference video's surf/LED        
        frame_rate  (int)           :   FPS of the video which df is based on #TODO: get this from video data  
        oneSec_bias (boolean)       :   True, if enabled the method will only consider those gating levels which have been hold for a minimum of 1 sec already 
                                        (as EXTD does as default with Elekta Linacs)  

    returns:
        dict : holding the mean, std of the estimated lower gating levels (both for LED and surf) inlcuding the estimated positions of these
    '''

    #iterate over data_ref
    # whenever scinti on, evaluate surf/LED postion. 
    # absolute smallest would also include beam-off latency region - can be avoided by only considering rows following < time of max std
    low_gat_lev_surf_all = []
    std_low_gat_lev_surf_all = []
    low_gat_lev_LED_all = []
    std_low_gat_lev_LED_all = []
    
    low_gat_lev_surf_this_cycle = df["px_roi_surf_normalized"].max()
    low_gat_lev_LED_this_cycle = df["y_com_LED_normalized"].max()
    std_low_gat_lev_surf_this_cycle = 100
    std_low_gat_lev_LED_this_cycle = 100

    #TODO: if df < 100, then stop: too few values with onesec vbias
    if oneSec_bias == True and df.shape[0] < frame_rate:
        sys.exit("Reference video shorter than one second - cannot apply bias and therefore cannot compute lower gting level. Exit.")

    idx = 121
    scinti_beenON = False    
    max_this_cycle = 0
    cycle_cnt = 0

    curr_pos_LED_mean_last_sec = df.loc[(idx-frame_rate):idx, "y_com_LED_normalized"].mean()
    while idx < total_num_frames_ref/sampling-2:
        while (curr_pos_LED_mean_last_sec > 0.3*max_this_cycle or not scinti_beenON) and idx < total_num_frames_ref/sampling-2: 
            #still in this cycle
            curr_pos_LED_mean_last_sec = df.loc[(idx-frame_rate):idx, "y_com_LED_normalized"].mean()
            scinti_state = df.loc[idx, "scinti_on"]
            if curr_pos_LED_mean_last_sec > max_this_cycle:
                max_this_cycle = curr_pos_LED_mean_last_sec
            if scinti_state:
                scinti_beenON = True
                mean_surf_last_sec = df.loc[(idx-frame_rate):idx, "px_roi_surf_normalized"].mean()
                std_mean_surf_last_sec = df.loc[(idx-frame_rate):idx, "px_roi_surf_normalized"].std()
                mean_LED_last_sec = df.loc[(idx-frame_rate):idx, "y_com_LED_normalized"].mean()
                std_mean_LED_last_sec = df.loc[(idx-frame_rate):idx, "y_com_LED_normalized"].std()

                if mean_surf_last_sec < low_gat_lev_surf_this_cycle:
                    low_gat_lev_surf_this_cycle = mean_surf_last_sec
                    #shortest time with lower gating level in this cycle:
                    std_low_gat_lev_surf_this_cycle = std_mean_surf_last_sec
                    #t_min_surf_last_sec = df[df["px_roi_surf_normalized"]==min_surf_last_sec]["frame_times"].min()
                if mean_LED_last_sec < low_gat_lev_LED_this_cycle:
                    #t_min_LED_last_sec = df[df["y_com_LED_normalized"]==min_LED_last_sec]["frame_times"].min()
                    low_gat_lev_LED_this_cycle = mean_LED_last_sec
                    std_low_gat_lev_LED_this_cycle = std_mean_LED_last_sec
            idx +=1

        #if curr_pos_LED_mean_last_sec < 0.3*max_this_cycle and scinti_beenON:
        #new cycle
        cycle_cnt +=1
        scinti_beenON = False
        low_gat_lev_surf_all.append(low_gat_lev_surf_this_cycle)
        low_gat_lev_LED_all.append(low_gat_lev_LED_this_cycle)
        std_low_gat_lev_surf_all.append(std_low_gat_lev_surf_this_cycle)
        std_low_gat_lev_LED_all.append(std_low_gat_lev_LED_this_cycle)
        low_gat_lev_surf_this_cycle = df["px_roi_surf_normalized"].max() #restart iteration
        low_gat_lev_LED_this_cycle = df["y_com_LED_normalized"].max() #restart iteration
        max_this_cycle = 0
        idx +=1
         
    low_gat_lat_lev = {"surf": {"mean":np.mean(low_gat_lev_surf_all), "std all":np.std(low_gat_lev_surf_all), "stds single": std_low_gat_lev_surf_all},
                        "LED":  {"mean":np.mean(low_gat_lev_LED_all), "std all":np.std(low_gat_lev_LED_all), "stds single": std_low_gat_lev_LED_all}
                        }
        
    return low_gat_lat_lev
"""
def find_lower_gating_level_from_reference(df_ref_incl_plateaus, df_ref_plateaus_stats):
    #now: group by scinti-cycle, find smallest plateau number and look up stats in plat_stats
    print(df_ref_plateaus_stats)

    mean_this_plateau_surf_each = []
    cnts_this_plateau_surf_each = []
    std_this_plateau_surf_each = []

    mean_below_plateau_surf_each = []
    cnts_below_plateau_surf_each = []
    std_below_plateau_surf_each = []

    low_gat_lvls_all_cycles = []
    low_gat_lvls_all_cycles_unc_L = []
    low_gat_lvls_all_cycles_unc_tot = []

    #only for comparison reasons:
    #medians_diff_all_cycles = []
    #---

    #iterate over each reference cycle (multiple stepper functions)
    for name, grp in df_ref_incl_plateaus.groupby('scinti_on_cycle', dropna=True):
        plateau_min_where_scinti_on = grp['plateauWpad'].min()

        # THIS plateau
        mean_this_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on, 'mean']
        std_this_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on, 'std']
        cnts_this_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on, 'count']
        sem_this_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on, 'SEM']
        ##add for total evaluation
        mean_this_plateau_surf_each.append(mean_this_plateau)
        std_this_plateau_surf_each.append(std_this_plateau)
        cnts_this_plateau_surf_each.append(cnts_this_plateau)

        # BELOW plateau
        mean_below_plateau =  df_ref_plateaus_stats.at[plateau_min_where_scinti_on-1.0, 'mean']
        std_below_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on-1.0, 'std']
        cnts_below_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on-1.0, 'count']
        sem_below_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on-1.0, 'SEM']
        ## add for total evaluation
        mean_below_plateau_surf_each.append(mean_below_plateau)
        std_below_plateau_surf_each.append(std_below_plateau)
        cnts_below_plateau_surf_each.append(cnts_below_plateau)

        #Delta
        delta_this_cycle = mean_this_plateau-mean_below_plateau
        std_delta_this_cycle = delta_this_cycle/np.sqrt(12)

        # lower gating level this scinti_cycle
        low_gat_lvl_this_cycle = (mean_this_plateau+mean_below_plateau)/2
        low_gat_lvl_this_cycle_unc_L = 1/2* np.sqrt(sem_this_plateau**2 + sem_below_plateau**2) #(sem_this_plateau+sem_below_plateau)
        low_gat_lvl_this_cycle_unc_tot = np.sqrt(low_gat_lvl_this_cycle_unc_L**2 + std_delta_this_cycle**2)

        low_gat_lvls_all_cycles.append(low_gat_lvl_this_cycle)
        low_gat_lvls_all_cycles_unc_L.append(low_gat_lvl_this_cycle_unc_L)
        low_gat_lvls_all_cycles_unc_tot.append(low_gat_lvl_this_cycle_unc_tot)

        # only for comparison: --
        ##median_this_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on, '50%']
        #median_below_plateau = df_ref_plateaus_stats.at[plateau_min_where_scinti_on-1.0, '50%']
        #medians_diff_all_cycles.append((median_this_plateau+median_below_plateau)/2)
        # ----

    print("low_gat_lvls_all_cycles: ", low_gat_lvls_all_cycles)
    print("low_gat_lvls_all_cycles_unc_L: ", low_gat_lvls_all_cycles_unc_L)
    print("low_gat_lvls_all_cycles_unc_tot: ", low_gat_lvls_all_cycles_unc_tot)
    
    # treat combined lower gating level as repeated measurements of single ones:
    low_gat_lvl_total = np.mean(low_gat_lvls_all_cycles)
    sqrd_sum_total_uncertainties = sum(i*i for i in low_gat_lvls_all_cycles_unc_tot)
    low_gat_lvl_unc_tot_total = (1/(len(low_gat_lvls_all_cycles_unc_tot))) * np.sqrt(sqrd_sum_total_uncertainties)

    '''
    # combined lower gating level:
    # p_bar_tot = 1/N_tot * sum (bar_p_k * N_k) for all k
    combined_mean_upper_plateaus = 1/(sum(cnts_this_plateau_surf_each)) * sum([bar_p_k*N_k for bar_p_k, N_k in zip(mean_this_plateau_surf_each, cnts_this_plateau_surf_each)])
    numerator_std_combo_up =  sum(((N_k - 1) * std_k**2) for N_k, std_k in zip(cnts_this_plateau_surf_each, std_this_plateau_surf_each))
    denominator_std_combo_up = sum((N_k - 1) for N_k in cnts_this_plateau_surf_each)
    combined_std_upper_plateaus = np.sqrt(numerator_std_combo_up / denominator_std_combo_up)
    combined_sem_upper_plateaus = combined_std_upper_plateaus/np.sqrt(sum(cnts_this_plateau_surf_each))

    combined_mean_lower_plateaus = 1/(sum(cnts_below_plateau_surf_each))* sum([bar_p_k*N_k for bar_p_k, N_k in zip(mean_below_plateau_surf_each, cnts_below_plateau_surf_each)])
    numerator_std_combo_lower = sum(((N_k -1) * std_k**2) for N_k, std_k in zip(cnts_below_plateau_surf_each, std_below_plateau_surf_each))
    denominator_std_combo_lower = sum((N_k-1) for N_k in cnts_below_plateau_surf_each)
    combined_std_lower_plateaus = np.sqrt(numerator_std_combo_lower/denominator_std_combo_lower)
    combined_sem_lower_plateaus = combined_std_lower_plateaus/np.sqrt(sum(cnts_below_plateau_surf_each))
    
    Delta_total = combined_mean_upper_plateaus - combined_mean_lower_plateaus
    std_Delta_total = Delta_total/np.sqrt(12)

    low_gat_lvl_total = (combined_mean_upper_plateaus+combined_mean_lower_plateaus)/2
    low_gat_lvl_unc_L_total = 1/2*np.sqrt(combined_sem_upper_plateaus**2 + combined_sem_lower_plateaus**2) #(combined_sem_upper_plateaus+combined_sem_lower_plateaus)
    low_gat_lvl_unc_tot_total = np.sqrt(low_gat_lvl_unc_L_total**2 + std_Delta_total**2) 

    print("low_gat_lvl_total: ", low_gat_lvl_total)
    print("low_gat_lvl_unc_tot_total:", low_gat_lvl_unc_tot_total)
    '''

    if allow_intermediate_plotting_of_reference:
        fig, (ax) = plt.subplots()
        ax.plot(df_ref_incl_plateaus['frame_times'], df_ref_incl_plateaus['px_roi_surf'], '.', color='orange', label = 'surf')
        ax.axhline(y = low_gat_lvl_total, color='k', linestyle='-', lw=1)
        ax.axhline(y = low_gat_lvl_total+low_gat_lvl_unc_tot_total, color='k', linestyle= '--', lw=1)
        ax.axhline(y = low_gat_lvl_total-low_gat_lvl_unc_tot_total, color='k', linestyle = '--', lw=1)
        #ax.axhline(y = median_av, color='red', linestyle = ':', lw=1)
        axb = ax.twinx()
        axc = ax.twinx()
        axc.plot(df_ref_incl_plateaus['frame_times'],df_ref_incl_plateaus['plateauWpad'], color = 'r', label = 'plateauWpad')
        axb.plot(df_ref_incl_plateaus['frame_times'],df_ref_incl_plateaus['scinti_on']*2, color = 'blue', label = 'scinti on')
        axb.plot(df_ref_incl_plateaus['frame_times'],df_ref_incl_plateaus['scinti_on_cycle'], label= 'cycle')
        ax.set_xlabel("time (ms)")
        plt.legend()
        plt.show()
    
    low_gat_lat_lvls = {"surf": {"total":low_gat_lvl_total,"unc_total":low_gat_lvl_unc_tot_total, "unc_single": low_gat_lvls_all_cycles_unc_tot}}
    return low_gat_lat_lvls

def make_full_exclude_list(list_of_list_of_exclusions):
    #list_of_list_of_exclusions = [[a,b], [c,d], ...]
    full_list = []
    for elem in list_of_list_of_exclusions:
        full_list = full_list + list(range(elem[0], elem[1]+1))
    return full_list

# ------------ Gating Latencies -----------------------#
def get_beamON_beamOFF_latencies(data_df, data_ref_df_plats, data_ref_df_stats, low_gat_lvls, reg, reg_exclude):
    #low_gat_lvls = {"surf": {"total":low_gat_lvl_total,"unc_total":low_gat_lvl_unc_tot_total, "unc_single": low_gat_lvls_all_cycles_unc_tot}}
    # TODO: output better as dataframe?

    '''
    data:               pandas df
    low_gat_lvls:        dict
    multi:              bool, if we expect more then one deep breath cycle -> more than one gating latency to measure
    #TODO: more than one breath cycle
    
    returns: beamON_lat, beamOFF_lat, tON_surf, tON_scinti, tOFF_surf, tOFF_scinti
    '''
    
    # if i want to modify the data frame, then i can use "data_df_mod" as variable name
    data_df_mod = data_df.copy()

    latencies = {}
    ltc = pd.DataFrame(columns = ['method', "scinti_on_cycle", "beamON_lat", 'beamON_lat_unc', "tON_surf", "tON_surf_unc", "tON_scinti", "tON_scinti_unc", "beamOFF_lat", "beamOFF_lat_unc", "tOFF_surf", "tOFF_surf_unc", "tOFF_scinti", "tOFF_scinti_unc"])
    ltc_last_indx = 0
    
    for method in ["surf"]:#["LED", "surf"]:
        low_gat_lvl = low_gat_lvls[method]["total"]
        low_gat_lvl_unc = low_gat_lvls[method]["unc_total"]

        print(low_gat_lvls)
        print("lower gating level surf set to: ", round(low_gat_lvl, 3), "\pm ", round(low_gat_lvl_unc,3))

        #1. create two new columns: df['<meth>_in_window'],
        meth_in_window = method +'_in_window'
        data_df_mod[meth_in_window] = False

        col_of_interest = "px_roi_surf" if method == "surf" else "y_com_LED"

        latencies[method] = {}
        
        idx_last_cycle = -1

        for scinti_on_cycle, grp in data_df_mod.groupby('scinti_on_cycle', dropna = True):
            print('------------------------------------')
            max_indx_this_cycle = grp.index.max()            
            # only uncomment for approach2:
            #if scinti_on_cycle == 1.0: 
                #idx_last_cycle+= max_indx_this_cycle
                #continue
            print('scinti cycle:', scinti_on_cycle)
            #TODO: This seems  to be very inefficient (at least I break at end of cycle...)
            # INFO: I am not only iterating in the grp. I am actually only using the grp to determne max_indx_this_cycle and then work with the indeces
            # better way #TODO: add breathing cycles! start at 0 and end it when scinti_on_cycle ends, then start a new one. then, I can iterate over breathing cycles.
           

            #init absurd values
            tON_surf = 0
            tON_scinti = 0
            tOFF_surf = 0
            tOFF_scinti = 0

            #tON_surf_fitting_region_start = max_indx_this_cycle # vice versa results
            #tON_surf_fitting_region_end = grp.index.min()
            #tOFF_surf_fitting_region_start = max_indx_this_cycle
            #tOFF_surf_fitting_region_end = grp.index.min()

            surfON_fitting_region_indx = []
            surfOFF_fitting_region_indx = []

            #init switches
            tON_surf_found = False
            tOFF_surf_found = False
            tON_scinti_found = False
            tOFF_scinti_found = False

            respective_breathing_cycle_df = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle)]
            #TODO: update loop below with this new df -does this work with keeping the indexes??

            # find tPN and tOFF
            for row in data_df_mod.itertuples(): #TODO: kann ich das hier eingraenzen auf zwischen min_indx und max_indx?
                if row.Index > idx_last_cycle: #idx_last_cycle:
                    ## this could be vectorized, if I'd vectorized scinti_found #TODO
                    ## then: 1) where within uncertainties AND scinti not been found yet? -> 2) smallest and largest index --> 3) get region
                    if (low_gat_lvl-low_gat_lvl_unc) <= data_df_mod.at[row.Index, col_of_interest] <= (low_gat_lvl+low_gat_lvl_unc):
                        if not tON_scinti_found:
                            #pdb.set_trace()
                            surfON_fitting_region_indx.append(row.Index)
                        else:
                            surfOFF_fitting_region_indx.append(row.Index)
                    ##-- to here --
                    ''' #former version:
                    if data_df_mod.at[row.Index, col_of_interest] >= (low_gat_lvl):
                        if not tON_surf_found:
                            #time in msec where roi_surface exceeding lower_gating_level
                            tON_surf= data_df_mod.at[row.Index, "frame_times"]
                            #switch found on:
                            tON_surf_found = True
                    #find first frame with tOFF_surf after tON_surf phase
                    if (tON_surf_found==True and (data_df_mod.at[row.Index, col_of_interest] < (low_gat_lvl))):
                        if not tOFF_surf_found:
                            tOFF_surf = data_df_mod.at[row.Index, "frame_times"]
                            tOFF_surf_found = True
                    '''
                    
                    if data_df_mod.at[row.Index, "scinti_on_cycle"] == scinti_on_cycle:
                        if not tON_scinti_found:
                            # time in msec where roi_scinti is >0
                            tON_scinti = data_df_mod.at[row.Index, "frame_times"]
                            #switch on:
                            tON_scinti_found = True
                    if (tON_scinti_found == True and data_df_mod.at[row.Index, "scinti_on_cycle"] != scinti_on_cycle):
                        if not tOFF_scinti_found:
                            tOFF_scinti = data_df_mod.at[row.Index, "frame_times"]
                            tOFF_scinti_found = True
                    if row.Index == max_indx_this_cycle+1:
                        idx_last_cycle = row.Index-1
                        break
             
            #approach1 -------------------------       
            #if fitting region is empty: take two closest values below and above gating level and interpolate between these, 
            #else: add them anyway

            #TODO: check if idxmax and idxmin are the right choice or if rather index.min und index.max!
            # lowerON = last data point below lower gating level during breath-in
            # upperON = first data point above gating level during breath-in
            # upperOFF= last data point above gating level during breath-out
            # lowerOFF= first data point below gating level during breath-out
            lowerON = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle) & (data_df_mod['scinti_on_cycle'].isna()) & (data_df_mod['px_roi_surf']<(low_gat_lvl-low_gat_lvl_unc))]['px_roi_surf'].index.max()
            upperON = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle) & (data_df_mod['scinti_on_cycle'].isna()) & (data_df_mod['px_roi_surf']>(low_gat_lvl+low_gat_lvl_unc))]['px_roi_surf'].index.min()

            upperOFF = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle) & (data_df_mod['scinti_on_cycle']==scinti_on_cycle) & (data_df_mod['px_roi_surf']>(low_gat_lvl+low_gat_lvl_unc))]['px_roi_surf'].index.max()
            lowerOFF = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle) & (data_df_mod['scinti_on_cycle']==scinti_on_cycle) & (data_df_mod['px_roi_surf']<(low_gat_lvl-low_gat_lvl_unc))]['px_roi_surf'].index.min()
                        
            surfON_fitting_region_indx.extend([lowerON,upperON])
            surfOFF_fitting_region_indx.extend([upperOFF,lowerOFF])
            #--------------------------------------------------------
            #approach4: fitting region by fixed numbers of points
            def define_fitting_region_by_fixed_numbers_of_datapoints(df, surf_region_of_interest, lowgatlvl, lowgatlvl_uncert, loc='surfon', numpts=10):
                
                if loc =="surfon":
                    indeces_where_value_above_lowgatlvl = [ind for ind in surf_region_of_interest.index if surf_region_of_interest[ind]>(lowgatlvl+lowgatlvl_uncert)]
                    indeces_where_value_below_lowgatlvl = [innd for innd in surf_region_of_interest.index if surf_region_of_interest[innd]<(lowgatlvl-lowgatlvl_uncert)]
                    #only keep the x smallest indeces above/below. If len<x then only keep len elements
                    fitting_region_below = indeces_where_value_below_lowgatlvl[-numpts:] if len(indeces_where_value_below_lowgatlvl) >= numpts else indeces_where_value_below_lowgatlvl
                    fitting_region_above = indeces_where_value_above_lowgatlvl[:numpts] if len(indeces_where_value_above_lowgatlvl) >= numpts else indeces_where_value_above_lowgatlvl

                if loc =="surfoff":
                    indeces_where_value_above_lowgatlvl = [ind for ind in surf_region_of_interest.index if surf_region_of_interest[ind]>(lowgatlvl+lowgatlvl_uncert)]
                    indeces_where_value_below_lowgatlvl = [innd for innd in surf_region_of_interest.index if surf_region_of_interest[innd]<(lowgatlvl-lowgatlvl_uncert)]
                    
                    if len(indeces_where_value_below_lowgatlvl)<=numpts:
                        my_new_below_list = [indeces_where_value_above_lowgatlvl[-1]] + indeces_where_value_below_lowgatlvl
                    #extend with 120 below anyhow:
                        e=1
                        while e <121:
                            my_new_below_list.append(max(my_new_below_list)+1)
                            e+=1
                        indeces_where_value_below_lowgatlvl = indeces_where_value_below_lowgatlvl+my_new_below_list 
                        #sort & eliminate duplicates
                        indeces_where_value_below_lowgatlvl = sorted(set(indeces_where_value_below_lowgatlvl))
                        print(indeces_where_value_below_lowgatlvl)

                        #only keep those below lowgatlvl-lowgatl_uncert
                        indeces_where_value_below_lowgatlvl = [ix for ix in indeces_where_value_below_lowgatlvl if data_df_mod.at[ix,'px_roi_surf']<(lowgatlvl-lowgatlvl_uncert)]

                    fitting_region_below = indeces_where_value_below_lowgatlvl[:numpts] if len(indeces_where_value_below_lowgatlvl)>= numpts else indeces_where_value_below_lowgatlvl
                    fitting_region_above = indeces_where_value_above_lowgatlvl[-numpts:] if len(indeces_where_value_above_lowgatlvl)>= numpts else indeces_where_value_above_lowgatlvl

                new_fit_region = fitting_region_below+fitting_region_above

                surf_fitting_region_indx = [min(new_fit_region), max(new_fit_region)]
                print(f"{loc}fitting_region in scinti cycle {scinti_on_cycle}: {surf_fitting_region_indx}")

                return surf_fitting_region_indx

            #approach3: extend to where it is still linear ---------------------------
            def extend_fitting_region_boundaries_basedon_detection_of_linear_region(surf_region_of_interest, before_region, lowgatlvl, lowgatlvl_uncert, data_ref_df_plats, data_ref_df_stats, scinticycle, windowMaxSmooth):
                #TODO: enable non-linear fitting, i.e. if not linear, do b-splines with within uncertainty region, then uncertainties increase though!
                #pdb.set_trace()

                local_maxima_indx_cands = scipy.signal.argrelmax(surf_region_of_interest.to_numpy(), order=int(windowMaxSmooth))[0]+surf_region_of_interest.index.min()
                #local_maxima_vals = surf_region_of_interest[local_maxima_indx]

                #keep only maxima within surf_region_of_interest.index (if data been excluded):
                local_maxima_indx = [i for i in local_maxima_indx_cands if i in surf_region_of_interest.index]

                local_max_indx_vals_pairs = list(zip(local_maxima_indx, surf_region_of_interest[local_maxima_indx]))
                #largest index of maximum above gating level
                filtered_pairs_above_low_gat_lvl = [(idxD, valD) for idxD, valD in local_max_indx_vals_pairs if valD >= (lowgatlvl+lowgatlvl_uncert)]
                if filtered_pairs_above_low_gat_lvl == []: #aka no local maxima above lower gating level
                    idx_first_max = surf_region_of_interest.idxmax()
                    filtered_pairs_above_low_gat_lvl = [(idx_first_max, surf_region_of_interest.at[idx_first_max])]
                local_maxima_above_low_gat_lvl_indx, local_maxima_above_low_gat_lvl_vals = zip(*filtered_pairs_above_low_gat_lvl)
                #print(local_maxima_above_low_gat_lvl_indx)

                #print(local_maxima_indx, '\n', local_maxima_vals)

                # set tolerance of residuals to 2 sigma of plateaus (take scinti on plateaus for reference to exclude the others)
                std_list = []
                for name, grp in data_ref_df_plats.groupby('scinti_on_cycle', dropna=True):
                    plateau_min_where_scinti_on = grp['plateauWpad'].min()
                    std_list.append(data_ref_df_stats.at[plateau_min_where_scinti_on, 'std'])

                tol_max = np.mean(std_list)*3 #TODO discuss this with GL and CK

                x = np.arange(min(before_region), max(before_region)+1, step=1)
                y = surf_region_of_interest[x].to_numpy()
                LG = scipy.stats.linregress(x, y)
                slope = LG.slope; intercept = LG.intercept; intercept_stderr = LG.intercept_stderr; stderr = LG.stderr
                y_regress = slope*x + intercept
                residuals = y - y_regress
                r_squared = LG.rvalue**2

                #initiation b and u 
                x_b = x; y_b = y; y_regress_b = y_regress; residuals_b = residuals; stderr_b = stderr;  r_squared_b = r_squared
                x_u = x;  y_u = y; y_regress_u = y_regress; residuals_u = residuals; stderr_u = stderr;  r_squared_u = r_squared

                #first below(surfON)/above(surfOFF)            
                x_cand_b = x_b
                y_cand_b = y_b
                y_regress_cand_b = y_regress_b
                residuals_cand_b = residuals_b
                stderr_cand_b = stderr_b
                r_squared_cand_b= r_squared_b

                #while r_squared_cand > 0.98:
                while np.mean((abs(residuals_cand_b)))<tol_max: #TODO: hier am besten die region anpassen, dass nicht immer ganzer Bereich angeschaut wird
                    x_b = x_cand_b
                    y_b = y_cand_b
                    y_regress_b = y_regress_cand_b
                    residuals_b = residuals_cand_b
                    stderr_cand_b = stderr_cand_b
                    r_squared_b = r_squared_cand_b

                    if min(x_b) == surf_region_of_interest.index.min(): 
                        #print(scinti_on_cycle, ': first half: broke because of minimum reached')
                        break
                    x_cand_b = np.sort(np.append(x_b,min(x_b)-1))
                    #if len(x_cand)> 100:
                        #print("broke due to length")
                     #   break

                    y_cand_b = surf_region_of_interest[x_cand_b].to_numpy()
                    if np.any(y_cand_b >= min(local_maxima_above_low_gat_lvl_vals)): 
                        #print(scinti_on_cycle, ': first half: broke because of maximum reached')
                        break

                    LG_cand_b= scipy.stats.linregress(np.array(x_cand_b),y_cand_b)
                    slope_cand_b = LG_cand_b.slope; intercept_cand_b = LG_cand_b.intercept; stderr_cand_b=LG_cand_b.stderr; intercept_stderr_cand_b = LG_cand_b.intercept_stderr
                    y_regress_cand_b = slope_cand_b*np.array(x_cand_b) + intercept_cand_b
                    if np.any(y_regress_cand_b >= min(local_maxima_above_low_gat_lvl_vals)): 
                        #print(scinti_on_cycle, ': first half: broke because of local max reached')
                        break
                    residuals_cand_b = y_cand_b - y_regress_cand_b
                    r_squared_cand_b = LG_cand_b.rvalue**2
                    #print(r_squared_cand_b)

                x_unten = x_b
                y_unten = y_b

                #then above(surfON)/below(surfOFF)
                x_cand_u= x_u
                y_cand_u = y_u
                y_regress_cand_u = y_regress_u
                residuals_cand_u = residuals_u
                stderr_cand_u = stderr_u
                r_squared_cand_u = r_squared_u
                #while r_squared_cand > 0.98: 
                while np.mean((abs(residuals_cand_u)))<tol_max:
                    #pdb.set_trace()
                    x_u = x_cand_u
                    y_u = y_cand_u
                    y_regress_u = y_regress_cand_u
                    residuals_u = residuals_cand_u
                    stderr_u = stderr_cand_u
                    r_squared_u = r_squared_cand_u
                    
                    if max(x_u) == surf_region_of_interest.index.max(): 
                        #print(scinti_on_cycle, ': second half: broke because of maximum reached')
                        break
                    x_cand_u = np.sort(np.append(x_u,max(x_u)+1))

                    y_cand_u = surf_region_of_interest[x_cand_u].to_numpy()
                    if np.any(y_cand_u >= min(local_maxima_above_low_gat_lvl_vals)): 
                        #print(scinti_on_cycle, ': second half: broke because of local max with y_cand reached')
                        break

                    LG_cand_u = scipy.stats.linregress(np.array(x_cand_u),y_cand_u)
                    slope_cand_u = LG_cand_u.slope; intercept_cand_u = LG_cand_u.intercept; stderr_cand_u=LG_cand_u.stderr; intercept_stderr_cand_u = LG_cand_u.intercept_stderr
                    y_regress_cand_u = slope_cand_u*np.array(x_cand_u) + intercept_cand_u      
                    if np.any(y_regress_cand_u >= min(local_maxima_above_low_gat_lvl_vals)): 
                        #print(scinti_on_cycle, ': second half: broke because of local max with regression reached')
                        break
                    residuals_cand_u = y_cand_u - y_regress_cand_u
                    r_squared_cand_u = LG_cand_u.rvalue**2
                    #print(r_squared_cand_u)
                
                #combine to one: 
                x_full = np.arange(min(x_b), max(x_u), step=1)
                y_full = surf_region_of_interest[x_full]
                fullregress = scipy.stats.linregress(x_full, y_full)
                y_full_regress = fullregress.slope*x_full + fullregress.intercept
                residuals_full = y_full_regress- y_full
                r_sqrt = np.mean(np.sqrt(residuals_full**2))
                #r_sqrt = np.mean(abs(residuals_full))

                #pre-final length
                print('pre-final len x: ' , len(x_full))

                #TODO: subtract end region, that are not fullfilling average, exclude beginning and end region where fits goodness is below r^2 = 0.99
                len_piece = 20 # but substract only 1
                
                # init
                r_sqrt_b = r_sqrt; r_sqrt_u = r_sqrt

                #from b:
                while r_sqrt_b > tol_max:
                    x_full = x_full[1:]
                    x_test = x_full[:len_piece]
                    y_test = surf_region_of_interest[x_test]
                    y_test_reg = fullregress.slope*x_test + fullregress.intercept
                    residuals_test = y_test - y_test_reg
                    r_sqrt_b = np.mean(np.sqrt(residuals_test**2))
                    #print(residuals_test)
                    #print(r_sqrt_u)
                    #print(len(x_full))

                #from u:
                while r_sqrt_u > tol_max:
                    #pdb.set_trace()
                    x_full = x_full[:-1] #subtract last data point from top
                    x_test = x_full[-len_piece:]
                    y_test = surf_region_of_interest[x_test]
                    y_test_reg = fullregress.slope*x_test +fullregress.intercept
                    residuals_test = y_test-y_test_reg
                    r_sqrt_u = np.mean(np.sqrt(residuals_test**2))
                    #print(residuals_test)
                    #print(r_sqrt_u)
                    #print(len(x_full))

                #after cutting
                print("len x after cutting: ", len(x_full))

                #keep it to minimum extend of "before_region"
                x_full_copy = x_full.copy()
                before_region_min_indx = min(before_region)
                before_region_max_indx = max(before_region)
                if (len(x_full_copy) == 0) or (min(x_full_copy) > before_region_min_indx):
                    #use len() since it is array or list
                    x_full = np.append(x_full, [before_region_min_indx])
                if (len(x_full_copy) == 0) or (max(x_full_copy) < before_region_max_indx):
                    x_full = np.append(x_full, [before_region_max_indx])

                #final length
                print('final len x: ' , len(x_full))

                #final regress:
                y_full = surf_region_of_interest[x_full]
                fullregress = scipy.stats.linregress(x_full, y_full)
                y_full_regress = fullregress.slope * x_full + fullregress.intercept
                residuals_full = y_full_regress -y_full

                #print(residuals_cand)
                #print('residuals_total', residuals)

                #for development 
                #print('residuals my fit: ', (abs(residuals)).mean())
                '''
                FLG = scipy.stats.linregress(surf_region_of_interest.index, surf_region_of_interest)
                FLG_y= FLG.slope*surf_region_of_interest.index +FLG.intercept

                x_lin = np.arange(min(before_region)-150, max(before_region), step=1) #surfON region
                #x_lin = np.arange(min(before_region), surf_region_of_interest.index.max()) #surfOFF region
                PLG = scipy.stats.linregress(x_lin, surf_region_of_interest[x_lin])
                PLG_y= PLG.slope*x_lin +PLG.intercept
                
                column_names = ['stderr', 'intercept err', 'mean residuals', 'r**2']
                index_names = ['lin fit', 'myfit', 'fullfit']
                df_lin_fit = pd.DataFrame(index=index_names, columns = column_names)
                df_lin_fit.loc['lin fit', 'stderr'] = PLG.stderr; df_lin_fit.loc['myfit', 'stderr'] = fullregress.stderr; df_lin_fit.loc['fullfit', 'stderr'] = FLG.stderr
                df_lin_fit.loc['lin fit', 'intercept err'] = PLG.intercept_stderr; df_lin_fit.loc['myfit', 'intercept err'] = fullregress.intercept_stderr; df_lin_fit.loc['fullfit', 'intercept err'] = FLG.intercept_stderr
                df_lin_fit.loc['lin fit', 'mean residuals'] = np.mean((abs(surf_region_of_interest[x_lin]-PLG_y))); df_lin_fit.loc['myfit', 'mean residuals'] = np.mean((abs(residuals_full))); df_lin_fit.loc['fullfit', 'mean residuals'] = np.mean((abs(surf_region_of_interest-FLG_y)))
                df_lin_fit.loc['lin fit', 'r**2'] = PLG.rvalue**2; df_lin_fit.loc['myfit', 'r**2'] = fullregress.rvalue**2; df_lin_fit.loc['fullfit', 'r**2'] = FLG.rvalue**2
                
                #print(df_lin_fit)
                print('tol max', tol_max)
                '''
                '''
                plt.close()
                plt.plot(surf_region_of_interest, '.', color = 'orange', label='surf data')
                plt.axhline(y=lowgatlvl+lowgatlvl_uncert, xmin=surf_region_of_interest.index.min(), xmax=surf_region_of_interest.index.max(), color='k', linestyle= '--')
                plt.axhline(y=lowgatlvl, xmin=surf_region_of_interest.index.min(), xmax=surf_region_of_interest.index.max(), color='k', label = 'low gat lvl')
                plt.axhline(y=lowgatlvl-lowgatlvl_uncert, xmin=surf_region_of_interest.index.min(), xmax=surf_region_of_interest.index.max(), color='k', linestyle= '--', label = "low gat lvl unc")
                
                #plt.plot(x_unten,y_unten,'*r', markersize=5, label= 'below')
                #plt.plot(x_u, y_u, '*r', label = "points reg u")
                plt.plot(x_u, y_regress_u, 'red', label = "regress u")
                #plt.plot(x_b, y_b, '*b', label = "point reg b")
                plt.plot(x_b, y_regress_b, 'b', label = "regress b")

                plt.plot(x_full, y_full_regress, 'm', label= 'my full regress')
                
                #plt.plot(surf_region_of_interest.index, FLG_y,'--g', label= 'regress of full surf region')
                #plt.plot(x_lin, PLG_y, '--c', label='regress known lin region')
                plt.legend()
                
                plt.show()
                '''
                return min(x_full), max(x_full)
            #-------------------------------------------------------------------------
            #approach2: extend to between local maxima---------------------
            ### breath_in region
            def extended_fitting_region_boundaries_basedon_locMax_for_surfON(dff, lowgatlvl, lowgatlvl_uncert, scinticycle, windowMaxSearch):
                #pdb.set_trace()
                # indices, values and local maxima/indeces of accent/breath-In region
                surf_sine_and_accent = dff[(dff['breathing_cycle']==scinticycle)& (dff['scinti_on_cycle'].isna())]['px_roi_surf']
                index_value_pairs_accent = [(index, row['px_roi_surf']) for index, row in dff[(dff['breathing_cycle']==scinticycle)& (dff['scinti_on_cycle'].isna())].iterrows()]

                local_maxima_increase_indx = scipy.signal.argrelmax(surf_sine_and_accent.to_numpy(), order=int(windowMaxSearch))[0]+surf_sine_and_accent.index.min()
                #print(local_maxima_increase_indx.shape)
                local_maxima_indx_value_pairs = list(zip(local_maxima_increase_indx, surf_sine_and_accent[local_maxima_increase_indx]))
                #print(local_maxima_increase_indx)
                
                ## Find the pair with the maximum value below lowgatlvl-0.66*diff(lowgatlvl,localmax)
                filtered_pairs_below_low_gat_lvl = [(indexB, valueB) for indexB, valueB in local_maxima_indx_value_pairs if valueB <= (lowgatlvl-lowgatlvl_uncert)]
                max_value_pair_below_low_gat_lvl = max(filtered_pairs_below_low_gat_lvl, key=lambda x: x[1], default=None)
                #pdb.set_trace()
                diff_lowgatlvl_to_maxvalbelow = (lowgatlvl-lowgatlvl_uncert)-max_value_pair_below_low_gat_lvl[1]
                if diff_lowgatlvl_to_maxvalbelow <0:
                    warnings.warn("Value warning -> fix your code!! /n The difference between the lower gating level and the smalles value above it is below Zero!", category=UserWarning)
                lower_boundary_val = (lowgatlvl-lowgatlvl_uncert)-0.66*diff_lowgatlvl_to_maxvalbelow
                # Filter pairs below lower_boundaries and find their max index
                filtered_pairs_below_fitregion= [(indexBB, valueBB) for indexBB, valueBB in index_value_pairs_accent if valueBB <= lower_boundary_val]
                max_indx_val_pair_below_boundary = max(filtered_pairs_below_fitregion, key=lambda x: x[0], default=None)
                max_indx_below_boundary = max_indx_val_pair_below_boundary[0]

                ## Find the pair with the minimum value above lowgatlvl+0.66*diff(lowgatlvl,localmax)
                filtered_pairs_above_low_gat_lvl = [(indexA, valueA) for indexA, valueA in local_maxima_indx_value_pairs if valueA >= (lowgatlvl+lowgatlvl_uncert)]
                min_value_pair_above_low_gat_lvl = min(filtered_pairs_above_low_gat_lvl, key=lambda y:y[1], default = None)
                diff_lowgatlvl_to_minvalabove = min_value_pair_above_low_gat_lvl[1]-(lowgatlvl+lowgatlvl_uncert)
                if diff_lowgatlvl_to_minvalabove < 0:
                    warnings.warn("Value warning -> fix your code!! /n The difference between the lower gating level and the highest value below it is below Zero!", category=UserWarning)
                upper_boundary_val = (lowgatlvl+lowgatlvl_uncert)+0.66*diff_lowgatlvl_to_minvalabove
                #Filter values above upper_boundary and find their min index
                
                filtered_pairs_above = [(indexAA, valueAA) for indexAA, valueAA in index_value_pairs_accent if valueAA >= upper_boundary_val]
                min_indx_val_pair_above_upper_boundary = min(filtered_pairs_above, key=lambda y: y[0], default=None)
                min_indx_above_upper_boundary = min_indx_val_pair_above_upper_boundary[0]

                return max_indx_below_boundary, min_indx_above_upper_boundary

            def extended_fitting_region_boundaries_basedon_locMax_for_surfOFF(dfff, lowgatlvl, lowgatlvl_uncert, scinticycle, windowMaxSearch):
                #TODO: better with extended breathing cycle!
                # indices, values and local maxima of decent/breath-out region
                surf_decent = dfff[(dfff['breathing_cycle']==scinticycle)& (dfff['scinti_on_cycle']==scinticycle)]['px_roi_surf'] #TODO after breathing_cycle extension extend this decent until end of cycle
                index_value_pairs_decent = [(index, row['px_roi_surf']) for index, row in data_df_mod[(dfff['breathing_cycle']==scinticycle)& (dfff['scinti_on_cycle']==scinticycle)].iterrows()]

                local_max_decent_indx = scipy.signal.argrelmax(surf_decent.to_numpy(), order=int(windowMaxSearch))[0]+surf_decent.index.min()
                if len(local_max_decent_indx) ==0:
                    local_max_decent_indx = [surf_decent.to_numpy().argmax(axis=0)+surf_decent.index.min()]
                local_max_decent_indx_vals_pairs = list(zip(local_max_decent_indx, surf_decent[local_max_decent_indx]))
                #largest index of maximum above gating level
                filtered_pairs_above_low_gat_lvl = [(idxD, valD) for idxD, valD in local_max_decent_indx_vals_pairs if valD >= (lowgatlvl+lowgatlvl_uncert)]
                max_idx_pair_of_local_maxima_above_low_gat_lvl = max(filtered_pairs_above_low_gat_lvl, key = lambda x: x[0], default=None)
                diff_lowgatlvl_to_maxabove = max_idx_pair_of_local_maxima_above_low_gat_lvl[1]-(lowgatlvl+lowgatlvl_uncert)
                if diff_lowgatlvl_to_maxabove < 0:
                    warnings.warn("Value warning -> fix your code!! /n The difference between the lower gating level and the highest value below it is below Zero!", category=UserWarning)
                upper_boundarysurfOFF_val = (lowgatlvl+lowgatlvl_uncert)+0.66*diff_lowgatlvl_to_maxabove
                #Find pair with highest index above this:
                filtered_pairs_above_upperBoundary = [(idxDD, valDD) for idxDD, valDD in index_value_pairs_decent if valDD >= upper_boundarysurfOFF_val]
                pair_with_max_idx_above_boundarysurfOFF = max(filtered_pairs_above_upperBoundary, key=lambda y: y[0], default = None)
                idx_of_pair_with_max_idx_above_boundarysurfOFF = pair_with_max_idx_above_boundarysurfOFF[0]

                idx_global_min_decent = np.argmin(surf_decent.to_numpy())+surf_decent.index.min() #TODO: check in with this if working on breathing cycles

                return idx_of_pair_with_max_idx_above_boundarysurfOFF, idx_global_min_decent

            def get_tON_surf_from_fit(fit_regionON,lowgatlvl,lowgatlvl_uncert, colortON,marksizetON):
                fit_regionON_df = data_df_mod.loc[min(fit_regionON):max(fit_regionON)]

                #make the fit
                tON_surf_region_fit_coeff, cov_tON = np.polyfit(fit_regionON_df['frame_times'], fit_regionON_df['px_roi_surf'], deg=1, full=False, cov = True) #including uncertainties
                t_tON_surf_fit_eval_grid = np.arange(fit_regionON_df['frame_times'].min(), fit_regionON_df['frame_times'].max(),step=0.1)
                slope_tON = tON_surf_region_fit_coeff[0]
                slope_tON_unc = cov_tON[0][0]
                const_tON = tON_surf_region_fit_coeff[1]
                const_tON_unc = cov_tON[1][1]

                tON_surf_fit = slope_tON*t_tON_surf_fit_eval_grid + const_tON
                #print("tON_surf_region_fit_coeff: \n",tON_surf_region_fit_coeff, "\n cov: \n", cov_tON)
                print(f'Slope surf ON (cycle{scinti_on_cycle}): {round(slope_tON,1)} \pm {round(cov_tON[0][0],3)} pts/ms')
                
                #find tON
                tON_s = (lowgatlvl-const_tON)/slope_tON
                
                #uncertainties as interception of fit and lowgatlvls_uncert (only considering the levels)
                tON_s_lower = ((lowgatlvl-lowgatlvl_uncert)-const_tON)/slope_tON
                tON_s_upper = ((lowgatlvl+lowgatlvl_uncert)-const_tON)/slope_tON

                #if only consider error in m, we have 4 worst case scenarios, which of two are mirrored due to Strahlensatz:
                abs_tON_lower_addUnc1 = abs(-lowgatlvl_uncert/(slope_tON+abs(slope_tON_unc))+tON_s-tON_s_lower)
                abs_tON_upper_addUnc1 = abs_tON_lower_addUnc1 # Strahlensatz
                abs_tON_lower_addUnc2 = abs(+lowgatlvl_uncert/(slope_tON-abs(slope_tON_unc))-tON_s+tON_s_lower)
                abs_tON_upper_addUnc2 = abs_tON_lower_addUnc2 # Strahlensatz
                
                tON_s_lower_uncert_sum = abs(tON_s-tON_s_lower)+(abs_tON_lower_addUnc1+abs_tON_lower_addUnc2)/2
                tON_s_upper_uncert_sum = abs(tON_s_upper-tON_s) + (abs_tON_upper_addUnc1+abs_tON_upper_addUnc2)/2

                #plot each on latency details plot
                if allow_intermediate_plotting_of_linearFits:
                    plt.errorbar(tON_s, lowgatlvl, yerr=None, xerr= [[tON_s_lower_uncert_sum], [tON_s_upper_uncert_sum]], fmt ='*', color = colortON, markersize = marksizetON,elinewidth=2, capsize=15 )
                    plt.plot(fit_regionON_df['frame_times'], fit_regionON_df['px_roi_surf'], '+', markersize = 5,  color = colortON)
                    plt.plot(t_tON_surf_fit_eval_grid, tON_surf_fit, color=colortON)

                return tON_s, tON_s_lower_uncert_sum, tON_s_upper_uncert_sum

            def get_tOFF_surf_from_fit(fit_regionOFF, lowgatlvl,lowgatlvl_uncert, colortOFF,marksizetOFF):     
                fit_regionOFF_df = data_df_mod.loc[min(fit_regionOFF):max(fit_regionOFF)]
                #make the fit
                tOFF_surf_region_fit_coeff, cov_tOFF = np.polyfit(fit_regionOFF_df['frame_times'], fit_regionOFF_df['px_roi_surf'], deg=1, full=False, cov = True)
                t_tOFF_surf_fit_eval_grid=np.arange(fit_regionOFF_df['frame_times'].min(), fit_regionOFF_df['frame_times'].max(),step=1) #ms steps
                slope_tOFF = tOFF_surf_region_fit_coeff[0]
                slope_tOFF_unc = cov_tOFF[0][0]
                const_tOFF = tOFF_surf_region_fit_coeff[1]
                const_tOFF_unc = cov_tOFF[1][1]
                
                tOFF_surf_fit = slope_tOFF*t_tOFF_surf_fit_eval_grid + const_tOFF
                print(f'Slope surf OFF (cycle{scinti_on_cycle}): {round(slope_tOFF,2)} \pm {round(cov_tOFF[0][0],3)} pts/ms')
                
                #find tOFF_surf 
                tOFF_s = (lowgatlvl-const_tOFF)/slope_tOFF

                #uncertainties as interception of fit and lowgatlvls_uncert:
                tOFF_s_lower = ((lowgatlvl-lowgatlvl_uncert)-const_tOFF)/slope_tOFF
                tOFF_s_upper = ((lowgatlvl+lowgatlvl_uncert)-const_tOFF)/slope_tOFF

                #if only consider error in m, we have 4 worst case scenarios, which of two are mirrored due to Strahlensatz:
                abs_tOFF_upper_addUnc2 = abs(-lowgatlvl_uncert/abs(slope_tOFF+slope_tOFF_unc)-tOFF_s + tOFF_s_upper)
                abs_tOFF_lower_addUnc2 = abs_tOFF_upper_addUnc2 #Strahlensatz
                abs_tOFF_upper_addUnc1 = abs(lowgatlvl_uncert/abs(slope_tOFF-slope_tOFF_unc) + tOFF_s - tOFF_s_upper)
                abs_tOFF_lower_addUnc1 = abs_tOFF_upper_addUnc1 # Strahlensatz
                
                tOFF_s_lower_uncert_sum = abs(tOFF_s - tOFF_s_lower)+(abs_tOFF_lower_addUnc2+abs_tOFF_lower_addUnc1)/2
                tOFF_s_upper_uncert_sum = abs(tOFF_s_upper - tOFF_s) + (abs_tOFF_upper_addUnc2+abs_tOFF_upper_addUnc1)/2

                if allow_intermediate_plotting_of_linearFits:
                    plt.plot(t_tOFF_surf_fit_eval_grid, tOFF_surf_fit, color=colortOFF)
                    plt.errorbar(tOFF_s, lowgatlvl, yerr=None, xerr= [[tOFF_s_lower_uncert_sum], [tOFF_s_upper_uncert_sum]],fmt ='*', color=colortOFF, markersize = marksizetOFF,elinewidth=2, capsize=15 )
                    plt.plot(fit_regionOFF_df['frame_times'], fit_regionOFF_df['px_roi_surf'], '+', markersize = 5,  color = colortOFF)
                    plt.plot(fit_regionOFF_df['frame_times'], fit_regionOFF_df['scinti_on'], color= utils.h_colorcodes.lmu_hellblau_100)
            
                return tOFF_s, tOFF_s_lower_uncert_sum, tOFF_s_upper_uncert_sum
            
            #smaller regions
            surfON_small = surfON_fitting_region_indx.copy()
            surfOFF_small = surfOFF_fitting_region_indx.copy()

            #extended regions
            #TODO: this is the line where my breathing cycles problem kicks in and I would assign empty breathing cycles to the following one which results into chaos...
            # remove these breathing cycles OR re-name breathing cycles to "beam cycles" and assign the empty parts to the cycle BEFORE them (then, analysis is already done before)
            # could also remove this cycles from analysis...?? 
            surfON_region  = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle)& (data_df_mod['scinti_on_cycle'].isna())]['px_roi_surf']
            surfOFF_region = data_df_mod[(data_df_mod['breathing_cycle']==scinti_on_cycle)& (data_df_mod['scinti_on_cycle']==scinti_on_cycle)]['px_roi_surf'] 
            #TODO after breathing_cycle extension extend this decent until end of cycle
            
            surfON_fitting_region_indx = define_fitting_region_by_fixed_numbers_of_datapoints(data_df_mod, surfON_region, low_gat_lvl, low_gat_lvl_unc, loc='surfon', numpts=10)
            surfOFF_fitting_region_indx = define_fitting_region_by_fixed_numbers_of_datapoints(data_df_mod, surfOFF_region, low_gat_lvl, low_gat_lvl_unc, loc= 'surfoff', numpts=10)

            '''
            lower_extended_bound, upper_extended_bound = extend_fitting_region_boundaries_basedon_detection_of_linear_region(surfON_region, surfON_small, low_gat_lvl, low_gat_lvl_unc, data_ref_df_plats, data_ref_df_stats,scinti_on_cycle, int(frame_rate*0.3))
            #lower_extended_bound, upper_extended_bound = extended_fitting_region_boundaries_basedon_locMax_for_surfON(data_df_mod, low_gat_lvl, low_gat_lvl_unc, scinti_on_cycle, frame_rate*0.3)
            #surfON_fitting_region_indx.extend([lower_extended_bound, upper_extended_bound])
            # 
            # lower_extended_bound_surfOFF, upper_extended_bound_surfOFF = extend_fitting_region_boundaries_basedon_detection_of_linear_region(surfOFF_region, surfOFF_small,low_gat_lvl, low_gat_lvl_unc, data_ref_df_plats, data_ref_df_stats,scinti_on_cycle,int(frame_rate*0.3))
            #lower_extended_bound_surfOFF, upper_extended_bound_surfOFF = extended_fitting_region_boundaries_basedon_locMax_for_surfOFF(data_df_mod, low_gat_lvl, low_gat_lvl_unc, scinti_on_cycle, frame_rate*0.3)
            #surfON_fitting_region_indx.extend([lower_extended_bound, upper_extended_bound])
            '''

            # with any size of linear fit region: small region
            #tON_surf, tON_surf_low_uncert, tON_surf_upper_uncert = get_tON_surf_from_fit(surfON_small, low_gat_lvl,low_gat_lvl_unc,utils.h_colorcodes.lmu_petrol_65,16)
            #tOFF_surf, tOFF_surf_low_uncert, tOFF_surf_upper_uncert = get_tOFF_surf_from_fit(surfOFF_small, low_gat_lvl,low_gat_lvl_unc,utils.h_colorcodes.lmu_hellgruen_100,16)

            #print('tON small: ', tON_surf, tON_surf_low_uncert, tON_surf_upper_uncert)
            #print('tOFF small: ', tOFF_surf, tOFF_surf_low_uncert, tOFF_surf_upper_uncert)
            
            # for comparison reasons: # with any size of linear fit region: large region
            tON_surf, tON_surf_low_uncert, tON_surf_upper_uncert = get_tON_surf_from_fit(surfON_fitting_region_indx, low_gat_lvl,low_gat_lvl_unc,utils.h_colorcodes.lmu_pink_100, 12)
            tOFF_surf, tOFF_surf_low_uncert, tOFF_surf_upper_uncert = get_tOFF_surf_from_fit(surfOFF_fitting_region_indx, low_gat_lvl,low_gat_lvl_unc,utils.h_colorcodes.lmu_dunkelgruen_100,12)
            #print('tON large: ', tON_surf, tON_surf_low_uncert, tON_surf_upper_uncert)
            #print('tOFF large: ', tOFF_surf, tOFF_surf_low_uncert, tOFF_surf_upper_uncert)
            
            # where tON_surf < time < tOFF_surf : meth_in_window = True
            #data_df_mod[meth_in_window] = (data_df_mod['frame_times'].between(tON_surf, tOFF_surf)).astype(bool)
            condition = ((data_df_mod['frame_times'] >= tON_surf) & (data_df_mod["frame_times"]<= tOFF_surf))
            data_df_mod.loc[condition, meth_in_window] = True

            #plt.plot(data_df_mod[data_df_mod['breathing_cycle']==scinti_on_cycle]['frame_times'], data_df_mod[data_df_mod['breathing_cycle']==scinti_on_cycle]['px_roi_surf'], '.')
            #plt.axhline(y= low_gat_lvls['surf']['total'], color='k', label='lowergatlvl')
            #plt.axhline(y = (low_gat_lvls['surf']['total']-low_gat_lvl_unc), color='k', linestyle= '--', label = 'lowergatlvl unc estimate?')
            #plt.axhline(y = (low_gat_lvls['surf']['total']+low_gat_lvl_unc), color='k', linestyle= '--', label = 'lowergatlvl unc estimate?')
            #plt.show()
            
            #pdb.set_trace()
            #latencies
            beamON_lat = tON_scinti - tON_surf
            beamOFF_lat = tOFF_scinti - tOFF_surf

            # compute and also return uncertainties
            tON_scinti_uncert =  (1000/frame_rate)/np.sqrt(3) #Delta between frames
            tOFF_scinti_uncert = (1000/frame_rate)/np.sqrt(3) #Delta between frames
            tON_surf_uncert = (tON_surf_low_uncert+ tON_surf_upper_uncert)/2 #but should be the same since linear fit comes with same horizontal distance when crossing same vertical distance
            tOFF_surf_uncert = (tOFF_surf_low_uncert+ tOFF_surf_upper_uncert)/2
            
            beamON_lat_uncert = np.sqrt(tON_scinti_uncert**2 + tON_surf_uncert**2)
            beamOFF_lat_uncert = np.sqrt(tOFF_scinti_uncert**2 + tOFF_surf_uncert**2)

            #test if reasonable #TODO add more
            if beamON_lat<0:
                warnings.warn("Physical error: Beam-on latency is smaller than 0 ms ", category=UserWarning)
            if beamOFF_lat<0:
                warnings.warn("Physical error: Beam-off latency is smaller than 0 ms ", category=UserWarning)
            if tOFF_scinti<tON_surf:
                warnings.warn("Your tOFF scinti is before tON surf. Check for: handling uncertainties? any bugs?")
            if tOFF_scinti<tOFF_surf:
                warnings.warn("Your tOFF scinti is before tOFF surface: Check for: handling uncertainties? any bugs?")
            if beamOFF_lat>beamON_lat:
                warnings.warn("beamOFF_lat>beamON_lat - This might be not okay- Check your code!")
            
            #write to dict
            ltc.loc[ltc_last_indx+1, 'method'] = method
            ltc.loc[ltc_last_indx+1, 'scinti_on_cycle'] = scinti_on_cycle

            ltc.loc[ltc_last_indx+1, 'beamON_lat'] = beamON_lat
            ltc.loc[ltc_last_indx+1, 'beamON_lat_unc'] = beamON_lat_uncert
            ltc.loc[ltc_last_indx+1, 'tON_surf'] = tON_surf
            ltc.loc[ltc_last_indx+1, 'tON_surf_unc'] = tON_surf_uncert
            ltc.loc[ltc_last_indx+1, 'tON_scinti'] = tON_scinti
            ltc.loc[ltc_last_indx+1, 'tON_scinti_unc'] = tON_scinti_uncert

            ltc.loc[ltc_last_indx+1, 'beamOFF_lat'] = beamOFF_lat
            ltc.loc[ltc_last_indx+1, 'beamOFF_lat_unc'] = beamOFF_lat_uncert
            ltc.loc[ltc_last_indx+1, 'tOFF_surf'] = tOFF_surf
            ltc.loc[ltc_last_indx+1, 'tOFF_surf_unc'] = tOFF_surf_uncert
            ltc.loc[ltc_last_indx+1, 'tOFF_scinti'] = tOFF_scinti
            ltc.loc[ltc_last_indx+1, 'tOFF_scinti_unc'] = tOFF_scinti_uncert
            ltc_last_indx +=1

            latencies[method][f'cycle{scinti_on_cycle}'] = {"beamON lat"   : {"result": beamON_lat,  "uncert": beamON_lat_uncert},   "beamOFF lat"  : {"result": beamOFF_lat,  "uncert": beamOFF_lat_uncert},
                                                            "beamON duration": {"result": (tOFF_scinti - tON_scinti), "uncert": np.sqrt(tOFF_scinti_uncert**2+tON_scinti_uncert**2)}, 
                                                        "tON surf"      : {"result": tON_surf,    "uncert": tON_surf_uncert},     "tOFF surf"    : {"result": tOFF_surf,    "uncert": tOFF_surf_uncert},
                                                        "tON scinti"    : {"result": tON_scinti,  "uncert": tON_scinti_uncert},   "tOFF scinti"  : {"result": tOFF_scinti,  "uncert": tOFF_scinti_uncert}
                                                        }
        #TODO überlegen of dies nicht besser in DataFrame könnte (auch für statistics etc.)
        # e.g. columns = method, scinti_on_cycle, beamON lat, mean, uncert, beamOFF lat, mean, uncert
    
    df_latencies_comparison_stats = add_statistics_for_development(ltc, cols_of_interest= ['beamON_lat', 'beamON_lat_unc', 'beamOFF_lat', 'beamOFF_lat_unc', 'tON_surf_unc', 'tOFF_surf_unc', 'duration surfON', 'duration surfON unc'])
    print(df_latencies_comparison_stats.astype(float).round(2))


    #plt.plot(data_df_mod['frame_times'],data_df_mod[meth_in_window]*data_df_mod['px_roi_surf'].max(), color=utils.h_colorcodes.lmu_gelb_100, label = f'{method} in window')
    plt.plot(data_df_mod['frame_times'],data_df_mod['px_roi_surf'], '.', markersize = 5,  color='orange', label = 'surface')
    plt.axhline(y= low_gat_lvls['surf']['total'], color='k', label='lowergatlvl')
    plt.axhline(y = (low_gat_lvls['surf']['total']-low_gat_lvl_unc), color='k', linestyle= '--', label = 'lowergatlvl unc')
    plt.axhline(y = (low_gat_lvls['surf']['total']+low_gat_lvl_unc), color='k', linestyle= '--', label = 'lowergatlvl unc')
    plt.plot(data_df_mod['frame_times'],data_df_mod['scinti_on']*data_df_mod['px_roi_surf'].max(), color= utils.h_colorcodes.lmu_hellblau_100, label = 'scinti on')
    plt.xlabel('time [ms]')
    plt.ylabel('ampl [px cnt]')
    plt.legend(loc = 'upper left')
    if allow_intermediate_plotting_of_std_curve_after_latencies:
        plt.show()
    return ltc, latencies, data_df_mod

def add_statistics_for_development(df1, cols_of_interest):
    df = df1.copy(deep=True)
    df['duration surfON'] = df['tOFF_surf']-df['tON_surf']
    df['duration surfON unc'] = df['tOFF_surf_unc']+df['tON_surf_unc']
    df=df[cols_of_interest]
    max_cycle = df.index.max()
    
    for col_name in cols_of_interest: 
        df.loc['mean', col_name] = df.loc[:max_cycle, col_name].mean()
        df.loc['std', col_name] = df.loc[:max_cycle,col_name].std()
        df.loc['min', col_name] = df.loc[:max_cycle,col_name].min()
        df.loc['max', col_name] = df.loc[:max_cycle,col_name].max()
    return df

def define_analysis_region(term, df, reg, reg_exclude):
    #choose part of video to analyze
    if reg == [[]]:
        print(f'Look at the following plot and note down, where the {term} region should start and end.')
        plt.plot(df['px_roi_surf'],  linestyle= ' ', marker='.', color='orange')
        plt.plot(df['px_roi_scinti']/df['px_roi_scinti'].max()*df['px_roi_surf'].max(), linestyle= ' ',  marker = '.', color= 'blue')
        plt.title(f'Define {term} region for {exp_note}.')
        plt.xlabel('frames [1]')
        plt.show()
        start_analysis_frame = int(input(f"At which frame does the {term} start?"))
        end_analysis_frame = int(input(f"At which frame does the {term} stop?"))
    else:
        start_analysis_frame = reg[0]
        end_analysis_frame = reg[1]
        print(f'start frame of {term}=', start_analysis_frame)
        print(f'last frame of {term}=',end_analysis_frame )
    
    if reg_exclude == [[]]:
        print(f'Do you want to exclude any regions {term}? Give region as list of list pairs, for example a,b:')        
        plt.plot(df['px_roi_surf'],  linestyle= ' ', marker='.', color='orange')
        plt.plot(df['px_roi_scinti']/df['px_roi_scinti'].max()*df['px_roi_surf'].max(), linestyle= ' ',  marker = '.', color= 'blue')
        plt.title(f'Define regions to be excluded from {term} for {exp_note}.')
        plt.xlabel('frames [1]')
        plt.show()

        lst=[]
        while True:
            inp=input("Give a region you want to exclude, e.g. a,b. Enter. If you don't want to add any new pairs, write 'stop':")
            if inp == 'stop':
                break
            else:
                inp_split = inp.split(',')
                lst.append(list(int(inp_split[0]),int(inp_split[1])))
        
        exclude_list = make_full_exclude_list(lst)
    elif reg_exclude ==[0]:
        exclude_list = []
    else:
        exclude_list = make_full_exclude_list(reg_exclude)

    # define rampup region
    df = df[start_analysis_frame:end_analysis_frame]
    df_copy = df.copy(deep=True)
    df = df[~df.index.isin(exclude_list)]
    df_no_analysis = df_copy[df_copy.index.isin(exclude_list)]

    return df, exclude_list, df_no_analysis

##### PLotting/export/Report Functions ##############
def numfmt_div1000(x, pos): # your custom formatter function: divide by 1000.0
    s = '{}'.format(x / 1000.0)
    return s

def make_final_figure():
    #minimum cycle with latencies:
    list_keys_dict_latencies = list(dict_latencies["surf"].keys())
    min_lat_cycle = list_keys_dict_latencies[0]

    # create figure
    fig, axs = plt.subplots(ncols=2, nrows=3,figsize=(8.2,11.6))# figsize = (8, 11.6))
    fig.suptitle(experiment)
    #ax1: image, ax2: reference surf
    #ax34=axbig: std curve full surf
    #axs[2,0]/5b: zoom beamON, axs[2,1]/ ax6b: zoom beamOFF
    
    #second row combined plot
    gs = axs[1,0].get_gridspec()
    for ax in axs[1,0:]: #remove underlying axes
        ax.remove()
    axbig = fig.add_subplot(gs[1,0:])

    #twin-axes for scintillator
    ax2b = axs[0,1].twinx()
    axbigb = axbig.twinx()
    ax5b = axs[2,0].twinx()
    ax6b = axs[2,1].twinx()

    axs[0,0].set_title("ROIs in respective analysis channel")
    # draw masked area on plotted frame
    if not skip_video: 
        frame_list = [plot_frame_col1, plot_frame_col3]
        for frame in frame_list:
            cv2.rectangle(frame,(roi_surface[0],roi_surface[1]), (roi_surface[2],roi_surface[3]), (255,255,0), 2 ) # (img, left upper corner, right lower corner, color, thickness)
            cv2.rectangle(frame, (roi_scinti[0], roi_scinti[1]), (roi_scinti[2], roi_scinti[3]), (255,255,0),2)

        axs[0,0].imshow(plot_frame_col1)

    if plot_ref_curve:
        axs[0,1].set_title('Reference motion')
        #TODO: plot excluded data: df_ref_no_analysis
        axs[0,1].plot(df_ref_no_analysis["frame_times"], df_ref_no_analysis["px_roi_surf"], '.', markersize=1, color=utils.h_colorcodes.lmu_gray_50, label = 'surface motion not analyzed')
        axs[0,1].plot(data_ref_mod["frame_times"] , data_ref_mod["px_roi_surf"], '.', markersize=1,color=surf_ref_color, label = 'surface motion')
        axs[0,1].axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k', label = "lower gat. level (-) & uncert.(--)")
        axs[0,1].axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        axs[0,1].axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        ax2b.plot(data_ref_mod["frame_times"] , data_ref_mod["scinti_on"],'-', markersize=1,color= scinti_ref_color, label = 'scintillator signal')
        #ax2b.plot(df_ref_no_analysis["frame_times"], df_ref_no_analysis["scinti_on"], '.', markersize=1, color=utils.h_colorcodes.lmu_hellblau_100, label = 'scinti not analyzed')
        for cycle, grp in data_ref_mod.groupby('scinti_on_cycle'):
            indx_min_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.min()
            indx_max_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.max()
            axs[0,1].add_patch(matplotlib.patches.Rectangle((data_ref_mod.at[indx_min_scinti_on,'frame_times'],data_ref_mod["px_roi_surf"].min()), 
                                            (data_ref_mod.at[indx_max_scinti_on,'frame_times']-data_ref_mod.at[indx_min_scinti_on,'frame_times']), data_ref_mod["px_roi_surf"].max()*1.01,
                                            facecolor = scinti_ref_color, alpha = 0.3, angle=0.0, rotation_point='xy', label = 'scintillator on'))
    # std motion and zoomies
    axbig.set_title('All DIBH cycles')
    axs[2,0].set_title(min_lat_cycle+': entry gating window')
    axs[2,1].set_title(min_lat_cycle+': exit gating window')
    for ax in [axbig, axs[2,0], axs[2,1]]:
        ax.plot(df_data_std_no_analysis["frame_times"], df_data_std_no_analysis["px_roi_surf"], '.', markersize=1, color=surf_exclude_color)
        ax.plot(data_mod["frame_times"] , data_mod["px_roi_surf"], '.', markersize=1,color=surf_color)#, label = 'surface')
        ax.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        for idx, row in df_latencies.iterrows():
                ax.add_patch(matplotlib.patches.Rectangle((row['tON_scinti'], data_mod["px_roi_surf"].min()),
                                            (row['tOFF_scinti']-row['tON_scinti']), data_mod["px_roi_surf"].max()*1.01,
                                            facecolor = scinti_color, alpha = 0.3, angle=0.0, rotation_point='xy'))            
    for axb in [axbigb, ax5b, ax6b]:
        axb.plot(data_mod["frame_times"] , data_mod["scinti_on"],'-', markersize=1,color= scinti_color)
        axb.plot(df_data_std_no_analysis["frame_times"], df_data_std_no_analysis["scinti_on"], '.', markersize=1, color=utils.h_colorcodes.lmu_hellblau_100, label = 'scinti not analyzed')

    if plot_gatlat:
    #TODO: add linear fits to zoomed in images?
        for ax in [axs[2,0], axs[2,1]]:
            #latencies
            for scintiON_cycle in dict_latencies["surf"].keys():
                ax.hlines(lower_gating_levels["surf"]["total"], xmin=(dict_latencies["surf"][scintiON_cycle]["tON surf"]["result"]) ,xmax=(dict_latencies["surf"][scintiON_cycle]["tON scinti"]["result"]) ,lw=4, color =utils.h_colorcodes.lmu_hellgruen_100, label = "beamON lat.")
                ax.hlines(lower_gating_levels["surf"]["total"], xmin=(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"]) ,xmax=(dict_latencies["surf"][scintiON_cycle]["tOFF scinti"]["result"]) ,lw=4, color= utils.h_colorcodes.lmu_dunkelgruen_100, label = "beamOFF lat.")
                ax.errorbar(dict_latencies["surf"][scintiON_cycle]["tON surf"]["result"], lower_gating_levels["surf"]["total"], 
                            yerr=None, xerr= [[dict_latencies["surf"][scintiON_cycle]["tON surf"]["uncert"]], [dict_latencies["surf"][scintiON_cycle]["tON surf"]["uncert"]]], 
                            fmt ='*', color='red', markersize =  1, elinewidth=1, capsize=2, label= "tON surf" )            
                ax.errorbar(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"], lower_gating_levels["surf"]["total"], 
                            yerr=None, xerr= [[dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]], [dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]]], 
                            fmt ='*', color='m', markersize =  1, elinewidth=1, capsize=2, label = "tOFF surf" )

        # zoomie beamON
        axs[2,0].set_ylim(ymin =(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]*10), ymax = data_mod['px_roi_surf'].max()*1.1)
        axs[2,0].set_xlim(xmin=(dict_latencies['surf'][min_lat_cycle]['tON surf']['result']-dict_latencies["surf"][min_lat_cycle]["tON surf"]["uncert"]*30), xmax = (dict_latencies['surf'][min_lat_cycle]['tON scinti']['result']+dict_latencies["surf"][scintiON_cycle]["tON scinti"]["uncert"]*30))# even closer
        
        # zoomie beamOFF
        axs[2,1].set_ylim(ymin =(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]*10), ymax = data_mod['px_roi_surf'].max()*1.1)
        axs[2,1].set_xlim(xmin=(dict_latencies['surf'][min_lat_cycle]["tOFF surf"]["result"]-dict_latencies["surf"][min_lat_cycle]["tOFF surf"]["uncert"]*30) , xmax=(dict_latencies['surf'][min_lat_cycle]["tOFF scinti"]["result"]+dict_latencies["surf"][min_lat_cycle]["tOFF scinti"]["uncert"]*30) )# even closer

    # axes labels and titles
    for ax_h in [ axs[0,1], axbig, axs[2,0], axs[2,1]]:
        ax_h.set_xlabel("time [ms]")
        ax_h.set_ylabel("amplitude [#pixels]")
    for ax_hB in [ax2b, ax5b, ax6b]:
        ax_hB.yaxis.set_ticks([0, 1], ["off", "on"])
        ax_hB.set_ylabel('scintillator binary [on/off]')

    ##generating custom legend
    handles_all = []
    labels_all = []
    patches_all = []
    for ax in [ axs[0,1], axbig, axs[2,0], axs[2,1], ax2b, axbigb, ax5b, ax6b]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in labels_all:
                handles_all.append(handle)
                labels_all.append(label)
                try: 
                    color_handle = handle.get_color()
                except:
                    try:
                        color_handle = handle[0].get_color() # e.g. for barcontainers.
                    except:
                        color_handle = handle.get_facecolor() # rectangles
                patches_all.append(matplotlib.patches.Patch(color=color_handle, label=label))
    fig.legend(handles = patches_all, labels = labels_all ,bbox_to_anchor=(0.58, 0.9), draggable=True)

    if show_final_plot:
        plt.show()
        
    if save_plots:
        fig.savefig(os.path.join(save_directory_experiment_path, "plot")+'_'+exp_note, format='png')

    return fig

def make_figure_abstract_ECMP2024():
    #for ECMP abstract
    
    list_keys_dict_latencies = list(dict_latencies["surf"].keys())
    min_lat_cycle = list_keys_dict_latencies[0]

    plt.rc('font', size=11)  


    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(8.2,11.6))# figsize = (8, 11.6))
    # ax00, ax01:           <empty/draggable legend> , reference
    # ax10, ax11 = axbig1   full row: all DIBH cycles
    # ax20, ax21:           ROI, exit gating window

    #make ax01 2/3 of the plot
    gs01 = axs[0,1].get_gridspec()
    for ax in axs[0,1:]:
        ax.remove()
    axbig0 = fig.add_subplot(gs01[0,1:])
    
    #make row2 full plot:    
    gs1 = axs[1,0].get_gridspec()
    for ax in axs[1,0:]:
        ax.remove()    
    axbig1 = fig.add_subplot(gs1[1,0:])

    #make ax21 2/3 of the plot
    gs21 = axs[2,1].get_gridspec()
    for ax in axs[2,1:]:
        ax.remove()
    axbig2 = fig.add_subplot(gs21[2,1:])

    #add twin-axes for scintillator:
    ax01b =axbig0.twinx()
    axbig1b = axbig1.twinx()
    ax21b = axbig2.twinx()

    # axs[0,0]
    axs[0,0].remove() #give room for legend

    #axs[0,1] reference motion
    axbig0.plot(data_ref_mod["frame_times"] , data_ref_mod["px_roi_surf"], '.', markersize=1,color=surf_ref_color, label = 'surface motion')
    axbig0.plot(df_ref_no_analysis["frame_times"], df_ref_no_analysis["px_roi_surf"], '.', markersize=1, color=surf_exclude_color, label = 'surface motion not analyzed')
    axbig0.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k', label = "lower gat. level (solid)")
    axbig0.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k', label = "lower gat. level uncert. (dashed)")
    axbig0.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
    for cycle, grp in data_ref_mod.groupby('scinti_on_cycle'):
        indx_min_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.min()
        indx_max_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.max()
        axbig0.add_patch(matplotlib.patches.Rectangle((data_ref_mod.at[indx_min_scinti_on,'frame_times'],data_ref_mod["px_roi_surf"].min()), 
                                        (data_ref_mod.at[indx_max_scinti_on,'frame_times']-data_ref_mod.at[indx_min_scinti_on,'frame_times']), data_ref_mod["px_roi_surf"].max()*1.01,
                                        facecolor = scinti_ref_color, alpha = 0.3, angle=0.0, rotation_point='xy', label = 'scintillator on'))
        #ax01b.plot(data_ref_mod["frame_times"] , data_ref_mod["scinti_on"],'-', markersize=1,color= scinti_ref_color, label = 'scintillator signal')
    axbig0.set_title('Reference motion')


    #axbig1, ax2,1:
    axbig1.set_title('All DIBH cycles')
    axbig2.set_title('DIBH cycle 1, exit gating window')
    for ax in [axbig1, axbig2]:
        #surface
        ax.plot(data_mod["frame_times"] , data_mod["px_roi_surf"], '.', markersize=1,color=surf_color)
        ax.plot(df_data_std_no_analysis["frame_times"], df_data_std_no_analysis["px_roi_surf"], '.', markersize=1, color=surf_exclude_color)
        #gating levels
        ax.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        #scintillator
        #ax.plot(data_mod["frame_times"] , data_mod["scinti_on"],'-', markersize=1,color= scinti_color)
        for idx, row in df_latencies.iterrows():
            ax.add_patch(matplotlib.patches.Rectangle((row['tON_scinti'], data_mod["px_roi_surf"].min()),
                                            (row['tOFF_scinti']-row['tON_scinti']), data_mod["px_roi_surf"].max()*1.01,
                                            facecolor = scinti_color, alpha = 0.3, angle=0.0, rotation_point='xy'))
        #latencies
        for scintiON_cycle in dict_latencies["surf"].keys():
            ax.hlines(lower_gating_levels["surf"]["total"], xmin=(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"]) ,xmax=(dict_latencies["surf"][scintiON_cycle]["tOFF scinti"]["result"]) ,lw=4, color= utils.h_colorcodes.lmu_dunkelgruen_100, label = "beamOFF latency")
            ax.errorbar(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"], lower_gating_levels["surf"]["total"], 
                        yerr=None, xerr= [[dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]], [dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]]], 
                        fmt ='*', color='m', markersize =  1, elinewidth=1, capsize=2, label = "tOFF surface" )        
        # zoomie beamOFF
        #TODO: mache erster scinti_on_cycle in df_latencies statt cyclex.0
        axbig2.set_xlim(xmin=(dict_latencies['surf'][min_lat_cycle]["tOFF surf"]["result"]-dict_latencies['surf'][min_lat_cycle]["beamOFF lat"]["result"]*0.6) , xmax=(dict_latencies['surf'][min_lat_cycle]["tOFF scinti"]["result"]+dict_latencies['surf'][min_lat_cycle]["beamOFF lat"]["result"]*0.6) )
        axbig2.set_ylim(ymin =(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]-data_mod['px_roi_surf'].max()*0.2), ymax = data_mod['px_roi_surf'].max()*1.2)

    #ax2,0
    if color_space == "BGR":
        axs[2,0].set_title("Video frame in blue channel") 
    elif color_space == "HSV":
        axs[2,0].set_title("Video frame in Hue") 
    # draw masked area on plotted frame
    if not skip_video: 
        frame_list = [plot_frame_col1, plot_frame_col3]
        for frame in frame_list:
            cv2.rectangle(frame,(roi_surface[0],roi_surface[1]), (roi_surface[2],roi_surface[3]), (255,255,0), 2 ) # (img, left upper corner, right lower corner, color, thickness)
            cv2.rectangle(frame, (roi_scinti[0], roi_scinti[1]), (roi_scinti[2], roi_scinti[3]), (255,255,0),2)

        axs[2,0].imshow(plot_frame_col1)

    # axes labels and titles
    for ax_h in [ axbig0, axbig1, axbig2]:
        ax_h.set_xlabel("time [ms]")
        ax_h.set_ylabel("amplitude [#pixels]")
    for ax_hB in [ax01b, axbig1b, ax21b]:
        ax_hB.yaxis.set_ticks([0, 1], ["off", "on"])
        ax_hB.set_ylabel('scintillator signal [on/off]')

    ##generating custom legend
    plt.rc('legend', fontsize=14) 
    handles_all = []
    labels_all = []
    patches_all = []
    for ax in [ axbig0, ax01b, axbig1, axbig1b, axbig2, ax21b]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in labels_all:
                handles_all.append(handle)
                labels_all.append(label)
                try: 
                    color_handle = handle.get_color()
                except:
                    try:
                        color_handle = handle[0].get_color() # e.g. for barcontainers.
                    except:
                        color_handle = handle.get_facecolor() # rectangles
                patches_all.append(matplotlib.patches.Patch(color=color_handle, label=label))
    fig.legend(handles = patches_all, labels = labels_all, draggable=True)
    fig.subplots_adjust(hspace=0.4,wspace=0.2)
    if show_final_plot:
        plt.show()

    return fig

def make_figures_ECMP2024():
    #for ECMP abstract
    
    list_keys_dict_latencies = list(dict_latencies["surf"].keys())
    min_lat_cycle = list_keys_dict_latencies[0]
    #print(list_keys_dict_latencies)
    #min_lat_cycle = 'cycle19.0'
    
    plt.rc('font', size=22)

    fig, ((axRef, axZoom),(axStd1, axStd2)) = plt.subplots(ncols=2, nrows=2, figsize=(34,12))# figsize = (8, 11.6))
    # ax00, ax01:           reference, detailed cycle (Lupe von zweiter Reihe)
    # ax10, ax11 = axbig1   full row: all DIBH cycles
    #legend: with arrows
    
    #make row2 full plot:    
    gs1 = axStd1.get_gridspec()
    for ax in [axStd1, axStd2]:
        ax.remove()    
    axStd = fig.add_subplot(gs1[1,0:])

    #add twin-axes for scintillator:
    axRefScinti = axRef.twinx()
    axStdScinti = axStd.twinx()
    axZoomScinti = axZoom.twinx()

    #reference motion
    #scintillator
    for cycle, grp in data_ref_mod.groupby('scinti_on_cycle'):
        indx_min_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.min()
        indx_max_scinti_on = data_ref_mod[data_ref_mod['scinti_on_cycle']==cycle].index.max()
        axRef.add_patch(matplotlib.patches.Rectangle((data_ref_mod.at[indx_min_scinti_on,'frame_times'],data_ref_mod["px_roi_surf"].min()), 
                                        (data_ref_mod.at[indx_max_scinti_on,'frame_times']-data_ref_mod.at[indx_min_scinti_on,'frame_times']), (data_ref_mod["px_roi_surf"].max()*1.01-data_ref_mod["px_roi_surf"].min()),
                                        facecolor = scinti_ref_color, alpha = 0.35, angle=0.0, rotation_point='xy', label = 'scintillator on'))
    #surface
    axRef.plot(data_ref_mod["frame_times"] , data_ref_mod["px_roi_surf"], '.', markersize=2,color=surf_ref_color, label = 'surface motion')
    axRef.plot(df_ref_no_analysis["frame_times"], df_ref_no_analysis["px_roi_surf"], '.', markersize=2, color=surf_exclude_color, label = 'surface motion not analyzed')
    axRef.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k', label = "lower gat. level (solid)")
    axRef.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k', label = "lower gat. level uncert. (dashed)")
    axRef.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
    axRef.set_title('Determination of lower gating level')

    #std
    axStd.set_title('DIBH breathing pattern')
    axZoom.set_title('DIBH cycle 15, exit gating window')
    for ax in [axStd, axZoom]:
        #scintillator
        for idx, row in df_latencies.iterrows():
            ax.add_patch(matplotlib.patches.Rectangle((row['tON_scinti'], data_mod["px_roi_surf"].min()),
                                            (row['tOFF_scinti']-row['tON_scinti']), (data_mod["px_roi_surf"].max()*1.01-data_mod["px_roi_surf"].min()),
                                            facecolor = scinti_color, alpha = 0.35, angle=0.0, rotation_point='xy'))
        #surface
        ax.plot(df_data_std_no_analysis["frame_times"], df_data_std_no_analysis["px_roi_surf"], '.', markersize=2, color=surf_exclude_color)
        #gating levels
        ax.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
        ax.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
    axStd.plot(data_mod["frame_times"] , data_mod["px_roi_surf"], '.', markersize=2,color=surf_color)
    #use bigger markers for zoom:
    axZoom.plot(data_mod["frame_times"] , data_mod["px_roi_surf"], '.', markersize=7,color=surf_color)
        #latencies
    for scintiON_cycle in dict_latencies["surf"].keys():
        axZoom.hlines(lower_gating_levels["surf"]["total"], xmin=(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"]) ,xmax=(dict_latencies["surf"][scintiON_cycle]["tOFF scinti"]["result"]) ,lw=5, color= beamOFF_lat_color, label = "beamOFF latency")
        axZoom.errorbar(dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["result"], lower_gating_levels["surf"]["total"], 
                    yerr=None, xerr= [[dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]], [dict_latencies["surf"][scintiON_cycle]["tOFF surf"]["uncert"]]], 
                    fmt ='*', color=beamOFF_lat_unc_color, markersize =  3, elinewidth=3, capsize=7, label = "tOFF surface" )        
    # zoomie beamOFF
        #TODO: mache erster scinti_on_cycle in df_latencies statt cyclex.0
    axZoom.set_xlim(xmin=(dict_latencies['surf'][min_lat_cycle]["tOFF surf"]["result"]-dict_latencies['surf'][min_lat_cycle]["beamOFF lat"]["result"]*0.7), xmax=(dict_latencies['surf'][min_lat_cycle]["tOFF scinti"]["result"]+dict_latencies['surf'][min_lat_cycle]["beamOFF lat"]["result"]*0.1) )
    axZoom.set_ylim(ymin =(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]-data_mod['px_roi_surf'].max()*0.05), ymax = (lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"])*1.03)
        
    # axes labels and titles
    for ax_h, ax_hB in zip([axRef, axStd, axZoom],[axRefScinti, axStdScinti, axZoomScinti]):
        ax_h.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(numfmt_div1000))
        ax_h.set_xlabel("time [s]")
        ax_h.set_ylabel("amplitude [#pixels]")
        ax_hB.yaxis.set_ticks([0,1], ["off", "on"])
        ax_hB.tick_params(axis='y', colors = utils.h_colorcodes.lmu_petrol_100)
    axStdScinti.set_ylabel('scintillator signal')#, color = utils.h_colorcodes.lmu_petrol_100)
    axZoomScinti.set_ylabel('scintillator signal')#, color = utils.h_colorcodes.lmu_petrol_100)
    
    #arrow annotations instead of legend
    # in axZoom
    xy_tip_tOFF = (dict_latencies["surf"][min_lat_cycle]["tOFF surf"]["result"],
                   lower_gating_levels["surf"]["total"]-10)
    xy_text_tOFF = (dict_latencies["surf"][min_lat_cycle]["tOFF surf"]["result"], 
                    lower_gating_levels["surf"]["total"]-130)
    xy_tip_beamOFFlat = (dict_latencies["surf"][min_lat_cycle]["tOFF surf"]["result"]+0.5*dict_latencies['surf'][min_lat_cycle]["beamOFF lat"]["result"],
                         lower_gating_levels["surf"]["total"]+10)
    xy_text_beamOFFlat = (0,40)# in textcoords = 'offset points'

    # in ref
    #surf signal
    xy_tip_surf = (35*1000,4000)
    xy_text_surf = (2*10,-45) #textcoords = 'offset points'
    #scintillator on (petrol), only textbox
    axRef.text(189.5*1000,3900,'scinti off',color = utils.h_colorcodes.lmu_petrol_100)
    axRef.text(224.5*1000,3900,'scinti on',color = utils.h_colorcodes.lmu_petrol_100)


    #the annotations:
    axZoom.annotate('tOFF,surface', xy=xy_tip_tOFF, xycoords = 'data', xytext=xy_text_tOFF, 
        arrowprops={"facecolor" : utils.h_colorcodes.lmu_dunkelgruen_100, "edgecolor" : 'none', 'width': 2, "shrink":1},
        color = utils.h_colorcodes.lmu_dunkelgruen_100
        )
    axZoom.annotate('beamOFF latency', xy=xy_tip_beamOFFlat, xycoords = 'data', textcoords = 'offset points', xytext=xy_text_beamOFFlat, 
        arrowprops={"facecolor" : utils.h_colorcodes.lmu_pink_100, "edgecolor" : 'none', 'width':2, "shrink":1},
        horizontalalignment="center", verticalalignment="center",
        color = utils.h_colorcodes.lmu_pink_100
        )
    axZoom.annotate('lower gating level \n & its uncertainties', xy = (536*1000,lower_gating_levels["surf"]["total"]),
                    xycoords = 'data', textcoords = 'offset points', xytext=(0,-50),
                    arrowprops = {"facecolor" : 'k', "edgecolor" : 'none', 'width':2, "shrink":1}, horizontalalignment="center", verticalalignment="center", color = 'k')
    axRef.annotate('surface signal', xy=xy_tip_surf, xycoords = 'data', textcoords = 'offset points', xytext=xy_text_surf, 
        arrowprops={"facecolor" : utils.h_colorcodes.lmu_orange, "edgecolor" : 'none', 'width': 2, "shrink":1},
        color = utils.h_colorcodes.lmu_orange
        )
    axRef.annotate('lower gating \n level', xy = (160*1000,(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]-20)),xycoords= 'data', textcoords ='offset points',
                    xytext = (-10,-40), arrowprops = {"facecolor" : 'k', "edgecolor" : 'none', 'width':2, "shrink":1}, horizontalalignment="center", verticalalignment="center", color = 'k')
    

    ##generating custom legend
    plt.rc('legend', fontsize=14) 
    handles_all = []
    labels_all = []
    patches_all = []
    for ax in [axRef, axRefScinti, axStd, axStdScinti]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in labels_all:
                handles_all.append(handle)
                labels_all.append(label)
                try: 
                    color_handle = handle.get_color()
                except:
                    try:
                        color_handle = handle[0].get_color() # e.g. for barcontainers.
                    except:
                        color_handle = handle.get_facecolor() # rectangles
                patches_all.append(matplotlib.patches.Patch(color=color_handle, label=label))
    #fig.legend(handles = patches_all, labels = labels_all, draggable=True)
    fig.subplots_adjust(hspace=0.3,wspace=0.4)

    ################## Video Frame ###########################
    frame_where_scinti_off = 0
    #in BGR:
    if not skip_video:
        plot_frame_scintiON_BGR = cv2.split(set_and_get_frame_of_interest(frame_where_scinti, fix_OS_specific_paths(vid1_file)))
        plot_frame_scintiON_blue = plot_frame_scintiON_BGR[0]
        plot_frame_scintiOff_BGR = cv2.split(set_and_get_frame_of_interest(frame_where_scinti_off, fix_OS_specific_paths(vid1_file)))
        plot_frame_scintiOff_blue = plot_frame_scintiOff_BGR[0]
    
    fig_frame, axs = plt.subplots(2,3,figsize=(16,6))
    # structure:
    # (axframe00,axframe01, axScintiOff, 
    # axframe10,axframe11, axScintiON)

    #combine first colum to one plot taking 2/3 of rows:
    gs = axs[0,0].get_gridspec()
    for ax in [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]:
        ax.remove()
    axframe = fig_frame.add_subplot(gs[0:,:-1])
    
    if not skip_video:
        imFrame = axframe.imshow(plot_frame_scintiON_blue, cmap='Blues', vmin=0, vmax=255)
        axs[1,2].imshow(plot_frame_scintiON_blue, cmap='Blues', vmin=0, vmax=255)
        axs[0,2].imshow(plot_frame_scintiOff_blue, cmap='Blues', vmin=0, vmax=255)
        for ax in [axframe, axs[0,2], axs[1,2]]:
            # draw masked area on plotted frame:
            ax.add_patch(matplotlib.patches.Rectangle((roi_surface[0],roi_surface[1]),(roi_surface[2]-roi_surface[0]), (roi_surface[3]-roi_surface[1]), 
                                                        linewidth=2, facecolor = 'none', edgecolor=utils.h_colorcodes.lmu_pink_100, 
                                                        rotation_point='xy', label = 'ROI'))
            ax.add_patch(matplotlib.patches.Rectangle((roi_scinti[0],roi_scinti[1]),(roi_scinti[2]-roi_scinti[0]), (roi_scinti[3]-roi_scinti[1]), 
                                                    linewidth=2, facecolor = 'none', edgecolor=utils.h_colorcodes.lmu_pink_100, 
                                                    rotation_point='xy', label = 'ROI'))

            ax.set_axis_off()

        axframe.annotate('ROI surface', xy=(roi_surface[2], roi_surface[1]+(roi_surface[3]-roi_surface[1])/3), xycoords = 'data', xytext=(roi_surface[3]+260,roi_surface[1]), 
                arrowprops={"facecolor" : utils.h_colorcodes.lmu_pink_100, "edgecolor" : 'none', "shrink":1},
                color = utils.h_colorcodes.lmu_pink_100
                )
        axframe.annotate('ROI scinti', xy=(roi_scinti[2], roi_scinti[3]-10), xycoords = 'data', xytext=(roi_scinti[2]+160,roi_scinti[3]+100),
                        arrowprops={"facecolor" : utils.h_colorcodes.lmu_pink_100, "edgecolor" : 'none', "shrink":1},
                        color = utils.h_colorcodes.lmu_pink_100
                        )
        
        scinti_zoom = [1261,1382,914,738]# (x1,x2,y1,y2)
        
        for axScinti in [axs[0,2], axs[1,2]]:
            axScinti.set_xlim([scinti_zoom[0], scinti_zoom[1]])
            axScinti.set_ylim([scinti_zoom[2],scinti_zoom[3]])
        
        #add texbox for scinti state
        axs[0,2].text(scinti_zoom[0]+5.5, scinti_zoom[2]-11,'scinti off',
                      bbox={'facecolor':'white', 'edgecolor':'k','boxstyle': 'square,pad=0.19'})
        axs[1,2].text(scinti_zoom[0]+6, scinti_zoom[2]-10.5,'scinti on',
                bbox={'facecolor':'white', 'edgecolor':'k', 'boxstyle': 'square,pad=0.22'})

        axframe.add_patch(matplotlib.patches.Rectangle((scinti_zoom[0], scinti_zoom[3]), (scinti_zoom[1]-scinti_zoom[0]), (scinti_zoom[2]-scinti_zoom[3]), # (x,y),w,h
                                        linewidth=2, facecolor = 'none', edgecolor='k', 
                                        rotation_point='xy'))
        #connect subplots:
        con1 = matplotlib.patches.ConnectionPatch(xyA=(scinti_zoom[1],scinti_zoom[3]), coordsA=axframe.transData,
                                                xyB=(scinti_zoom[0],scinti_zoom[3]), coordsB=axs[0,2].transData)        
        con2 = matplotlib.patches.ConnectionPatch(xyA=(scinti_zoom[1],scinti_zoom[2]), coordsA=axframe.transData,
                                                xyB=(scinti_zoom[0],scinti_zoom[2]), coordsB=axs[1,2].transData)
        fig_frame.add_artist(con1)
        fig_frame.add_artist(con2)

        fig_frame.subplots_adjust(wspace=-0.5)
        fig_frame.colorbar(imFrame, ax=axs[0:,2], location = 'right')

    return fig, fig_frame

def export_df_to_word_table(df, memo, stats=True, borders=False):
    import docx

    dff = df.copy(deep=True)
    dff = dff.drop('method', axis=1)
    
    if stats:
        max_cycle = dff.index.max()
        for col_name in dff.columns: 
            dff.loc['mean', col_name] = dff.loc[:max_cycle, col_name].mean()
            dff.loc['std', col_name] = dff.loc[:max_cycle,col_name].std()
            dff.loc['min', col_name] = dff.loc[:max_cycle,col_name].min()
            dff.loc['max', col_name] = dff.loc[:max_cycle,col_name].max()

    #round results
    dff = dff.astype(float).round(1)
    # Initialise the Word document
    doc = docx.Document()
    # Initialise the table
    t = doc.add_table(rows=(dff.shape[0] + 1), cols=dff.shape[1])
    if borders:
        t.style = 'TableGrid'
    # Add the column headings
    for j in range(dff.shape[1]):
        t.cell(0, j).text = dff.columns[j]
    # Add the body of the data frame
    for i in range(dff.shape[0]):
        for j in range(dff.shape[1]):
            cell = dff.iat[i, j]
            t.cell(i + 1, j).text = str(cell)
    # Save the Word doc
    filename = os.path.join(save_directory_experiment_path,'table_'+memo)
    doc.save(filename+'.docx')


################################################################################################
######################   DATA & INPUT DATA #####################################################
################################################################################################

with open(args.experimentFile, 'r') as f:
    exp_info = json.load(f)

# General Experiment Information
year_exp = exp_info['experimentGeneralInfo']['yearOfExperiment']
mon_exp = exp_info['experimentGeneralInfo']['monthOfExperiment']
day_exp = exp_info['experimentGeneralInfo']['dayOfExperiment']
room_mode = exp_info['experimentGeneralInfo']['roomAmbience']
exp_note = exp_info['experimentGeneralInfo']['noteExperiment']

# Motion Details
ampl_max_std = exp_info["DataInput"]["motion"]["ampl_max_std"]
ampl_min_std = exp_info["DataInput"]["motion"]["ampl_min_std"]
length_refPlateaus = exp_info["DataInput"]["motion"]["length_plateaus_ref"]

# General Video Information
skip_video = exp_info["DataInput"]['videos']['skip_video']
vid1_file= exp_info["DataInput"]['videos']['video1']['filename']
vid1_region = exp_info["DataInput"]['videos']['video1']['region']
vid1_region_exclude = exp_info["DataInput"]['videos']['video1']['exclude']

vid_ref1_file = exp_info["DataInput"]['videos']['video ref 1']['filename']
vid_ref1_region = exp_info["DataInput"]['videos']['video ref 1']['region']
vid1_ref1_region_exclude = exp_info["DataInput"]['videos']['video ref 1']['exclude']

new_GoProVideo = exp_info["DataInput"]['videos']['video1']['newGoProVideo']
new_Video= exp_info["DataInput"]['videos']['video1']['newVideo']
new_RefGoProVideo = exp_info["DataInput"]['videos']['video ref 1']['newGoProVideo']
new_RefVideo = exp_info["DataInput"]['videos']['video ref 1']['newVideo']

frame_rate = exp_info["DataInput"]['videos']['video1']['FPS']

# values for post-processing
max_gap_in_beam_analysis_seconds = exp_info["post-processingSettings"]["max_gap_in_beam_analysis_seconds"]
max_gap_in_beam_analysis_pts = max_gap_in_beam_analysis_seconds*frame_rate
max_gap_in_beam_ref_seconds = exp_info["post-processingSettings"]["max_gap_in_beam_ref_seconds"]
max_gap_in_beam_ref_pts = max_gap_in_beam_ref_seconds*frame_rate

# analysis settings
LED_tracking = exp_info["analysisSettings"]["flags"]["LED_tracking"]

load_from_postprocessedData = exp_info["analysisSettings"]["runtime"]["load_from_post-processedData"]
loop_over_reference = exp_info["analysisSettings"]["runtime"]["loop_over_reference"]
loop_over_video = exp_info["analysisSettings"]["runtime"]["loop_over_video"]
sampling =  exp_info["analysisSettings"]["runtime"]["sampling"]

roi_surface = set_boundaries_roi(exp_info['analysisSettings']["ROIs"]['roi_surface'][0],
                                 exp_info['analysisSettings']["ROIs"]['roi_surface'][1],
                                 exp_info['analysisSettings']["ROIs"]['roi_surface'][2],
                                 exp_info['analysisSettings']["ROIs"]['roi_surface'][3]
                                 )
roi_surface_ref = roi_surface
roi_LED = set_boundaries_roi(exp_info['analysisSettings']["ROIs"]['roi_LED'][0],
                             exp_info['analysisSettings']["ROIs"]['roi_LED'][1],
                             exp_info['analysisSettings']["ROIs"]['roi_LED'][2],
                             exp_info['analysisSettings']["ROIs"]['roi_LED'][3]
                             )
roi_scinti = set_boundaries_roi(exp_info['analysisSettings']["ROIs"]['roi_scinti'][0],
                                exp_info['analysisSettings']["ROIs"]['roi_scinti'][1],
                                exp_info['analysisSettings']["ROIs"]['roi_scinti'][2],
                                exp_info['analysisSettings']["ROIs"]['roi_scinti'][3]
                                )
color_space =  exp_info["analysisSettings"]["thresholds"]["color_space"]
if color_space =="BGR":
    threshold_surf = exp_info["analysisSettings"]["thresholds"]["threshold_surface"]
    threshold_surf_norm = exp_info["analysisSettings"]["thresholds"]["threshold_surface_normalized"]
elif color_space == "HSV":
    threshold_surf = exp_info["analysisSettings"]["thresholds"]["threshold_surface_hue"]
    threshold_surf_norm = exp_info["analysisSettings"]["thresholds"]["threshold_surface_normalized_hue"]
threshold_scinti = exp_info["analysisSettings"]["thresholds"]["threshold_scintillator"]
threshold_LED = exp_info["analysisSettings"]["thresholds"]["threshold_LED"]

roi_thresholds_set = exp_info["analysisSettings"]["flags"]["rois_and_thresholds_set"]


#controls
frame_where_scinti = exp_info["controlSettings"]["frame_where_scintillator_on"]
show_binary = exp_info["controlSettings"]["show_binary"]

#Developer Settings
allow_intermediate_plotting_of_reference = exp_info["developerSettings"]["intermediate_plotting_of_reference"]
allow_intermediate_plotting_of_linearFits = exp_info["developerSettings"]["intermediate_plotting_of_linearFits"]
allow_intermediate_plotting_of_std_curve_after_latencies = exp_info["developerSettings"]["allow_intermediate_plotting_of_std_curve_after_latencies"]
show_final_plot = exp_info["developerSettings"]["show_final_plot"]
keep_debugger_at_the_end = exp_info["developerSettings"]["keep_debugger_trace_at_end"]

# post-processing settings
comp_lat =  exp_info["post-processingSettings"]["compute_latency"]
comp_lat_wLED = exp_info["post-processingSettings"]["compute_latency_by_LED"]

# save Settings
save_postprocessed_results = exp_info["post_post-processingSettings"]["saveSettings"]["save_postprocessed_results"]
save_latency_results = exp_info["post_post-processingSettings"]["saveSettings"]["save_latency_results"]
save_plots = exp_info["post_post-processingSettings"]["saveSettings"]["save_plots"]
report_PDF = exp_info["post_post-processingSettings"]["saveSettings"]["create_PDFreport"]

#plotting
plot_gatlat =  exp_info["post_post-processingSettings"]["plotSettings"]["plot_gatlat"]
plot_std_curve = exp_info["post_post-processingSettings"]["plotSettings"]["plot_std_curve"]
plot_ref_curve = exp_info["post_post-processingSettings"]["plotSettings"]["plot_reference_curve"]
plot_LED = exp_info["post_post-processingSettings"]["plotSettings"]["plot_LED"]

#save directory
save_directory = exp_info["post_post-processingSettings"]["saveSettings"]["save_directory"]


##########################################################################################################
#####################      START MAIN / PRE PROCESSING      ##############################################
##########################################################################################################

experiment = fr"{year_exp}{mon_exp}{day_exp}_{room_mode}_std_{frame_rate}FPS_surf{threshold_surf[0]}to{threshold_surf[1]}_s{sampling}_{exp_note}"
exp_reference = fr"{year_exp}{mon_exp}{day_exp}_{room_mode}_a1_{frame_rate}FPS_surf{threshold_surf[0]}to{threshold_surf[1]}_s{sampling}_{exp_note}"


save_directory_experiment_path = fix_OS_specific_paths(os.path.join(save_directory, experiment))
if not os.path.exists(save_directory_experiment_path):
    os.makedirs(save_directory_experiment_path)

#save the config file
try:
    shutil.copy2(args.experimentFile, (os.path.normpath(os.path.join(save_directory_experiment_path,args.experimentFile))))
except:
    shutil.copy2(args.experimentFile, (os.path.normpath(os.path.join(save_directory_experiment_path,"config.json"))))

# filenames
filename= os.path.join(save_directory_experiment_path,'data_points_std')+'_'+exp_note+'.txt'
ref_filename = os.path.join(save_directory_experiment_path,'data_points_ref')+'_'+exp_note+'.txt' 

filename_postprocessed_data = (os.path.join(save_directory_experiment_path,'data_points_postprocessed_std')+'_'+exp_note+os.path.extsep+'txt')
filename_postprocessed_data_ref = (os.path.join(save_directory_experiment_path,'data_points_postprocessed_ref')+'a1'+ '_'+exp_note+os.path.extsep+'txt')
filename_postprocessed_excluded_data = (os.path.join(save_directory_experiment_path,'excluded_data_points_postprocessed_std')+'_'+exp_note+os.path.extsep+'txt')
filename_postprocessed_excluded_ref = (os.path.join(save_directory_experiment_path,'excluded_data_points_postprocessed_ref')+'a1'+ '_'+exp_note+os.path.extsep+'txt')

filename_dict_latencies_save = save_directory_experiment_path +os.path.sep + "dict_latencies"+'_'+exp_note+os.path.extsep+'txt'
filename_latencies = save_directory_experiment_path +os.path.sep + "df_latencies"+'_'+exp_note+os.path.extsep+'txt'
filename_latencies_stats = save_directory_experiment_path +os.path.sep + "df_latencies_stats"+'_'+exp_note+os.path.extsep+'txt'

filename_total_results = (os.path.join(save_directory_experiment_path,'results')+'_'+exp_note+os.path.extsep+'txt')
filename_lower_gating_levels = (os.path.join(save_directory_experiment_path,'dict_lower_gating_levels')+'_'+exp_note+os.path.extsep+'txt')

#plot frames
if vid1_file !=0:
    plot_frame = set_and_get_frame_of_interest(frame_where_scinti, fix_OS_specific_paths(vid1_file))
    if color_space == "BGR":
        plot_frame = plot_frame
    elif color_space =="HSV":
        plot_frame = cv2.cvtColor(plot_frame, cv2.COLOR_BGR2HSV)
    plot_frame_col1, plot_frame_col2, plot_frame_col3 = cv2.split(set_and_get_frame_of_interest(frame_where_scinti, fix_OS_specific_paths(vid1_file)))

############## Start redo analysis #####################################
if not load_from_postprocessedData: #REDO ANALYSIS
    
    # Convert to Os-specific paths and create if needed
    if vid1_file != 0:
        vid1 =          fix_OS_specific_paths(vid1_file)
    if vid_ref1_file != 0:
        vid_ref1=       fix_OS_specific_paths(vid_ref1_file)

    ############ start video pre-processing ###########################

    # parameter dependencies
    if not comp_lat:
        plot_gatlat = False

    if not skip_video:
        roi_list = [roi_surface, roi_surface_ref, roi_scinti]
        if LED_tracking: roi_list.append(roi_LED)

        # if new GoPro Video: load and copy with moviepy
        if new_GoProVideo:
            vid = copy_input_video(vid1)
        else:
            vid = vid1
            if new_Video:
                vid = cut_video(vid)

        if not vid1_file == vid_ref1_file:
            if new_RefGoProVideo:
                vid_ref1_cutcopy = copy_input_video(vid_ref1)
            else: 
                vid_ref1_cutcopy = vid_ref1
                if new_Video:
                    vid_ref1_cutcopy = cut_video(vid_ref1)
        else: 
            vid_ref1_cutcopy = vid_ref1

        ### Initialize data
        #read in video
        cap = cv2.VideoCapture(vid)
        cap_ref = cv2.VideoCapture(vid_ref1_cutcopy)

        ### print properties of video
        print_data_of_interest_from_cap(cap,vid, new_GoProVideo_flag=new_GoProVideo)
        print_data_of_interest_from_cap(cap_ref, vid_ref1_cutcopy, new_GoProVideo_flag=new_GoProVideo)

        #run through the video and create data
        total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_num_frames_ref1 = cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)

        # only if ROI/threshold not set OR for experimenting on a single frame
        if not roi_thresholds_set: 
            foi = set_and_get_frame_of_interest(frame_where_scinti,vid)
            #state, this_frame = cap.read()

            gray = cv2.cvtColor(foi,cv2.COLOR_BGR2GRAY)
            b,g,r = cv2.split(foi)
            hsv_frame = cv2.cvtColor(foi,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv_frame)
            

            # adding rectangles to only the frames to be shown
            for roi in roi_list: #roi_list =[roi_surface, roi_surface_ref, roi_scinti, roi_LED]
                cv2.rectangle(b,(roi[0],roi[1]), (roi[2],roi[3]), (255,0,255), 2 )
                cv2.rectangle(r,(roi[0],roi[1]), (roi[2],roi[3]), (255,0,255), 2 )
                cv2.rectangle(foi,(roi[0],roi[1]), (roi[2],roi[3]), (255,0,255), 2 )
            
            plt.imshow(cv2.cvtColor(foi, cv2.COLOR_BGR2RGB))
            plt.show()

            fig, ((ax1,ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize = (15,7))
            ax1.imshow(gray)
            ax1.set_title('gray channel - choose thresholds and ROIs')
            ax2.imshow(b)
            ax2.set_title('blue channel - choose thresholds and ROIs')

            ax3.imshow(gray/np.average(gray))
            ax3.set_title("gray scale - averaged over gray channel frame")
            ax4.imshow(b/np.average(b))
            ax4.set_title('blue channel - averaged over blue channel frame')

            ax5.imshow(hsv_frame, cmap='hsv')
            ax5.set_title("HSV")
            ax6.imshow(h, cmap='hsv')
            ax6.set_title("hue of HSV")
            plt.show()

            #plot histogram of scinti region: HSV: WIP
            #make_hist_im_stack([foi], roi_scinti, "HSV")
            #histHSV = cv2.calcHist([hsv_frame], [0,1,2], None, [182,256,256], [0,256])
            #plt.plot(histHSV)
            #make_hist_im_stack([foi], roi_scinti, "RGB")
            #histRGB = cv2.calcHist([cv2.cvtColor(foi, cv2.COLOR_BGR2RGB)], [0,1,2], None, [256,256,256], [0,256])
            #plt.plot(histRGB)

            sys.exit()

    ##################################################################################################
    ########################             PROCESSING               ####################################
    ##################################################################################################

        if loop_over_video:
            sample = 0
            
            stat, start_frame = cap.read()

            # first frame gray and color space
            start_frame_gray = cv2.cvtColor(start_frame,cv2.COLOR_BGR2GRAY)
            if color_space == "BGR":
                start_frame = start_frame
            elif color_space == "HSV":
                start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2HSV)
            
            #BGR: 1=blue, 2=green, 3 = red; HSV: 1=hue, 2=saturation, 3=value
            start_frame_col1, start_frame_col2, start_frame_col3 = cv2.split(start_frame)

            frame_count = 1
            frame_cnts_list = [frame_count]
            frame_times_list = [0]

            #average values
            frame_gray_av_list = [np.average(start_frame_gray)]
            frame_col1_av_list = [np.average(start_frame_col1)]
            frame_col2_av_list = [np.average(start_frame_col2)]
            frame_col3_av_list = [np.average(start_frame_col3)]

            roi_gray_av_list = [analyze_average_roi(start_frame_gray, roi_surface)]
            roi_col1_av_list = [analyze_average_roi(start_frame_col1, roi_surface)]
            roi_col2_av_list = [analyze_average_roi(start_frame_col2, roi_surface)]
            roi_col3_av_list = [analyze_average_roi(start_frame_col3, roi_surface)]

            # first values 
            px_roi_surf_nextVal, roi_surf_size = analyze_px_roi_surf(start_frame_col1, threshold_surf, threshold_surf_norm, roi_surface,color_space)
            px_roi_surf = [px_roi_surf_nextVal] # surface
            roi_surf_size_list = [roi_surf_size]

            px_roi_surf_nextVal_normFrame = analyze_px_roi_surf(start_frame_col1, threshold_surf, threshold_surf_norm, roi_surface, color_space)[0]
            px_roi_surf_normFrame = [px_roi_surf_nextVal_normFrame]
            
            px_roi_scinti_nextVal, roi_scinti_size= analyze_px_roi_scinti(start_frame_col1, threshold_scinti, roi_scinti, color_space) #scintillator
            px_roi_scinti = [px_roi_scinti_nextVal]
            roi_scinti_size_list = [roi_scinti_size]

            if LED_tracking:
                x_pos_com_LED_nextVal, y_pos_com_LED_nextVal = find_center_LED(start_frame_col3, threshold_LED, roi_LED)
                x_pos_com_LED_list = [x_pos_com_LED_nextVal]
                y_pos_com_LED_list = [y_pos_com_LED_nextVal]

            #little progress bar:
            bar = Bar('Processing Video Frames: ', max = total_num_frames)


            while True:
                #read next frame
                ret, frame = cap.read()
                sample+=1
                if not ret: break

                if sample % sampling == 0:
                    # get frame count and time
                    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_time_msec = 1000*frame_count/(cap.get(cv2.CAP_PROP_FPS))
                    frame_times_list.append(frame_time_msec)
                    frame_cnts_list.append(frame_count)

                    # channelling frame
                    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    if color_space == "BGR":
                        frame = frame
                    elif color_space == "HSV":
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    frame_onlyCol1, frame_onlyCol2, frame_onlyCol3 = cv2.split(frame)

                    #append px data
                    px_roi_surf_nextVal, roi_surf_size = analyze_px_roi_surf(frame_onlyCol1, threshold_surf, threshold_surf_norm, roi_surface, color_space)
                    px_roi_surf.append(px_roi_surf_nextVal) #surface
                    roi_surf_size_list.append(roi_surf_size)

                    px_roi_surf_normFrame.append(analyze_px_roi_surf(frame_onlyCol1, threshold_surf, threshold_surf_norm, roi_surface, color_space)[0])

                    px_roi_scinti_nextVal, roi_scinti_size= analyze_px_roi_scinti(frame_onlyCol1, threshold_scinti, roi_scinti,color_space) #scintillator
                    px_roi_scinti.append(px_roi_scinti_nextVal)
                    roi_scinti_size_list.append(roi_scinti_size)

                    # get LED position
                    if LED_tracking:
                        x_pos_com_LED_nextVal, y_pos_com_LED_nextVal = find_center_LED(frame_onlyCol3, threshold_LED, roi_LED)
                        x_pos_com_LED_list.append(x_pos_com_LED_nextVal)
                        y_pos_com_LED_list.append(y_pos_com_LED_nextVal)

                    frame_gray_av_list.append(np.average(frame_gray))
                    frame_col1_av_list.append(np.average(frame_onlyCol1))
                    frame_col2_av_list.append(np.average(frame_onlyCol2))
                    frame_col3_av_list.append(np.average(frame_onlyCol3))

                    roi_gray_av_list.append(analyze_average_roi(frame_gray, roi_surface))
                    roi_col1_av_list.append(analyze_average_roi(frame_onlyCol1, roi_surface))
                    roi_col2_av_list.append(analyze_average_roi(frame_onlyCol2, roi_surface))
                    roi_col3_av_list.append(analyze_average_roi(frame_onlyCol3, roi_surface))
                        
                #progress bar
                bar.next()
            cap.release()
            print("\n Analysis of standard video completed!\n saving data...")

            # write data to a dataframe and .txt
            d={'frame_cnts':frame_cnts_list, 'frame_times': frame_times_list, 
            'frame_gray_av':frame_gray_av_list, 'frame_col1_av': frame_col1_av_list, 'frame_col2_av': frame_col2_av_list, 'frame_col3_av': frame_col3_av_list,
            'roi_surf_gray_av': roi_gray_av_list, 'roi_surf_col1_av': roi_col1_av_list, 'roi_surf_col2_av': roi_col2_av_list, 'roi_surf_col3_av': roi_col3_av_list,   
            'px_roi_surf': px_roi_surf, 'px_roi_scinti': px_roi_scinti, 'px_roi_surf_normFrame': px_roi_surf_normFrame, 'roi_surf_size': roi_surf_size_list, 'roi_scinti_size': roi_scinti_size_list}
            if LED_tracking:
                d['x_com_LED']=x_pos_com_LED_list
                d['y_com_LED']=y_pos_com_LED_list 
            df = pd.DataFrame(data=d)
            df.to_csv(filename,index=False) #default: sep=',' delimiter='\n'
            print("DataFrame of standard video written!")

        if loop_over_reference:
            if vid_ref1_file == vid1_file:
                df_refX = df
                ref_filename = filename 
                print("DataFrame of reference video taken from video1.")
            else:
                sample = 0
                
                stat, start_frame_ref = cap_ref.read()

                # first frame
                start_frame_ref_gray = cv2.cvtColor(start_frame_ref,cv2.COLOR_BGR2GRAY)
                if color_space == "BGR":
                    start_frame_ref = start_frame_ref
                elif color_space == "HSV":
                    start_frame_ref = cv2.cvtColor(start_frame_ref, cv2.COLOR_BGR2HSV)

                start_frame_ref_col1, start_frame_ref_col2, start_frame_ref_col3 = cv2.split(start_frame_ref)

                frame_count_ref = 1
                frame_cnts_list_ref = [frame_count_ref]
                frame_times_list_ref = [0]

                # first values 
                px_roi_surf_ref_nextVal, roi_surf_size_ref = analyze_px_roi_surf(start_frame_ref_col1, threshold_surf,threshold_surf_norm,  roi_surface, color_space)
                px_roi_surf_ref = [px_roi_surf_ref_nextVal] # surface
                roi_surf_size_list_ref = [roi_surf_size_ref]

                px_roi_surf_ref_normFrame = [analyze_px_roi_surf(start_frame_ref_col1, threshold_surf,threshold_surf_norm, roi_surface, color_space)[0]]

                px_roi_scinti_ref_nextVal, roi_scinti_size_ref= analyze_px_roi_scinti(start_frame_ref_col1,threshold_scinti, roi_scinti,color_space) #scintillator
                px_roi_scinti_ref = [px_roi_scinti_ref_nextVal]
                roi_scinti_size_list_ref = [roi_scinti_size_ref]

                if LED_tracking:
                    x_pos_com_LED_ref_nextVal, y_pos_com_LED_ref_nextVal = find_center_LED(start_frame_ref_col3, threshold_LED, roi_LED)
                    x_pos_com_LED_ref_list = [x_pos_com_LED_ref_nextVal]
                    y_pos_com_LED_ref_list = [y_pos_com_LED_ref_nextVal]

                frame_ref_gray_av_list = [np.average(start_frame_ref_gray)]
                frame_ref_col1_av_list = [np.average(start_frame_ref_col1)]
                frame_ref_col2_av_list = [np.average(start_frame_ref_col2)]
                frame_ref_col3_av_list = [np.average(start_frame_ref_col3)]

                roi_ref_gray_av_list = [analyze_average_roi(start_frame_ref_gray, roi_surface)]
                roi_ref_col1_av_list = [analyze_average_roi(start_frame_ref_col1, roi_surface)]
                roi_ref_col2_av_list = [analyze_average_roi(start_frame_ref_col2, roi_surface)]
                roi_ref_col3_av_list = [analyze_average_roi(start_frame_ref_col3, roi_surface)]


                #little progress bar:
                bar = Bar('Processing Reference Video Frames: ', max = total_num_frames_ref1)

                while True:
                    #read next frame
                    ret, frame_ref = cap_ref.read()
                    sample+=1
                    if not ret: break

                    if sample % sampling == 0:
                        # get frame count and time
                        frame_count_ref = cap_ref.get(cv2.CAP_PROP_POS_FRAMES)
                        frame_time_msec_ref = 1000*frame_count_ref/(cap_ref.get(cv2.CAP_PROP_FPS))
                        frame_times_list_ref.append(frame_time_msec_ref)
                        frame_cnts_list_ref.append(frame_count_ref)

                        # grayscale image for surface, blue channel only for scinti
                        frame_ref_gray = cv2.cvtColor(frame_ref,cv2.COLOR_BGR2GRAY)
                        frame_ref_onlyCol1, frame_ref_onlyCol2, frame_ref_onlyCol3 = cv2.split(frame_ref)

                        #append px data
                        px_roi_surf_ref_nextVal, roi_surf_size = analyze_px_roi_surf(frame_ref_onlyCol1, threshold_surf, threshold_surf_norm,roi_surface, color_space)
                        px_roi_surf_ref.append(px_roi_surf_ref_nextVal) #surface
                        roi_surf_size_list_ref.append(roi_surf_size_ref)

                        px_roi_surf_ref_normFrame.append(analyze_px_roi_surf(frame_ref_onlyCol1, threshold_surf,threshold_surf_norm, roi_surface, color_space)[0])

                        px_roi_scinti_ref_nextVal, roi_scinti_size= analyze_px_roi_scinti(frame_ref_onlyCol1, threshold_scinti, roi_scinti,color_space) #scintillator
                        px_roi_scinti_ref.append(px_roi_scinti_ref_nextVal)
                        roi_scinti_size_list_ref.append(roi_scinti_size)

                        # get LED position
                        if LED_tracking:
                            x_pos_com_LED_ref_nextVal, y_pos_com_LED_ref_nextVal = find_center_LED(frame_ref_onlyCol3, threshold_LED, roi_LED)
                            x_pos_com_LED_ref_list.append(x_pos_com_LED_ref_nextVal)
                            y_pos_com_LED_ref_list.append(y_pos_com_LED_ref_nextVal)

                        frame_ref_gray_av_list.append(np.average(frame_ref_gray))
                        frame_ref_col1_av_list.append(np.average(frame_ref_onlyCol1))
                        frame_ref_col2_av_list.append(np.average(frame_ref_onlyCol2))
                        frame_ref_col3_av_list.append(np.average(frame_ref_onlyCol3))

                        roi_ref_gray_av_list.append(analyze_average_roi(frame_ref_gray, roi_surface))
                        roi_ref_col1_av_list.append(analyze_average_roi(frame_ref_onlyCol1, roi_surface))
                        roi_ref_col2_av_list.append(analyze_average_roi(frame_ref_onlyCol2, roi_surface))
                        roi_ref_col3_av_list.append(analyze_average_roi(frame_ref_onlyCol3, roi_surface))
                            
                        if show_binary:
                            list_small_outlier = [100.0, 200.0]
                            list_strong_outlier = []
                            if (frame_count_ref in list_small_outlier or frame_count_ref in[x-1 for x in list_small_outlier] or frame_count_ref in [y+1 for y in list_small_outlier]
                                                    or frame_count_ref in list_strong_outlier or frame_count_ref in [zz-1 for zz in list_strong_outlier] or frame_count_ref in [z+1 for z in list_strong_outlier]):
                                print(f"\n This is frame_cnt {frame_count_ref}:")
                                frameOI = np.where(((frame_ref_onlyCol1 > threshold_surf[0])&(frame_ref_onlyCol1 < threshold_surf[1])),255,0)
                                cv2.rectangle(frameOI,(roi_surface[0],roi_surface[1]), (roi_surface[2],roi_surface[3]), (0,255,255), 2 )
                                plt.imshow(frameOI)
                                plt.title(frame_count_ref)
                                
                                framenorm = frame_ref_onlyCol1/np.average(frame_ref_onlyCol1)
                                frameOInorm = np.where(((framenorm > threshold_surf_norm[0])&(framenorm < threshold_surf_norm[1])), 255, 0)
                                cv2.rectangle(frameOInorm,(roi_surface[0],roi_surface[1]), (roi_surface[2],roi_surface[3]), (0,255,255), 2 )
                                plt.imshow(frameOInorm)
                                plt.title(frame_count_ref)

                                plt.show()

                    #progress bar
                    bar.next()
                cap_ref.release()
                print("\n Analysis of reference video completed!\n saving data...")

                # write data to a dataframe and .txt
                d_refX = {'frame_cnts':frame_cnts_list_ref, 'frame_times': frame_times_list_ref, 
                        'frame_gray_av':frame_ref_gray_av_list, 'frame_col1_av': frame_ref_col1_av_list, 'frame_col2_av': frame_ref_col2_av_list, 'frame_col3_av': frame_ref_col3_av_list,
                        'roi_surf_gray_av': roi_ref_gray_av_list, 'roi_surf_col1_av': roi_ref_col1_av_list, 'roi_surf_col2_av': roi_ref_col2_av_list, 'roi_surf_col3_av': roi_ref_col3_av_list,  
                        'px_roi_surf': px_roi_surf_ref, 'px_roi_scinti': px_roi_scinti_ref, 'px_roi_surf_ref_normFrame': px_roi_surf_ref_normFrame,'roi_surf_size': roi_surf_size_list_ref, 'roi_scinti_size': roi_scinti_size_list_ref }
                if LED_tracking:
                    d_refX['x_com_LED']=x_pos_com_LED_ref_list
                    d_refX['y_com_LED']=y_pos_com_LED_ref_list 
                df_refX = pd.DataFrame(data=d_refX)
                df_refX.to_csv(ref_filename,index=False) #default: sep=',' delimiter='\n'
                
                print("DataFrame of reference video saved to file.")

    ##############################################################
    ######## POST PROCESSING - ESTIMATE GATING LATENCIES #########
    ##############################################################
    # min-max normalization data
    def min_max_normalized_dataFrame(df, cols_do, cols_not):
        for col in df.columns:
            if col not in cols_do:
                if col in cols_not:
                    df[col] = df[col]#*(-1)
                new_col_name = col +'_normalized'
                df[new_col_name] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
        return df

    # get data from txt (reference following further down)

    data = pd.read_csv(filename)

    print('Reading std data from file: ', filename)
    print("Reading std video data from file: \n", data)


    #min_max normalization of specific columns TODO: still active for scintillator!
    data = min_max_normalized_dataFrame(data, 
                                        cols_do =["frame_cnts", "frame_times", "frame_gray_av", "roi_surf_size", "roi_scinti_size"], 
                                        cols_not = ["x_com_LED", "y_com_LED"])

    # add scintillator True/False
    add_scinti_state_to_df(data,0.4)
    #TODO: this takes a long while
    print('scintillator state added')

    # identify regions with scinti being on 
    df_data = add_scinti_cycle_to_df(data, max_gap_in_beam_analysis_pts)
    #TODO: this takes a while
    print('scintillator cycles added')

    # add breathing cycles to data std
    add_breathing_cycles(df_data)
    print('breathing cycles added')

    #exclude regions #TODO: geht das hier hinten? wenn nicht, dasnn muss ich es vorziehen und wieder einzeln über ref und std rüber iterieren....
    df_data_std, final_excluded_region_vid1, df_data_std_no_analysis = define_analysis_region("analysis of latencies", df_data, vid1_region, vid1_region_exclude)

    if vid1_file == vid_ref1_file: # if std and reference in same video
        ref_data = data.copy()
        df_ref = add_scinti_cycle_to_df(ref_data, max_gap_in_beam_ref_pts)
        print('scintillator cycles added to reference')
        add_breathing_cycles(df_ref)
        print('breathing cycles added to reference')
        df_ref, final_excluded_region_vid_ref1, df_ref_no_analysis = define_analysis_region("reference" , df_ref, vid_ref1_region, vid1_ref1_region_exclude)
    else:
        #do it all for df_ref# get reference data from txt
        ref_filename_load = os.path.join(save_directory_experiment_path,'data_points_ref')+'a1'+'_'+exp_note+'.txt' 
        data_ref = pd.read_csv(ref_filename_load)
        print('Reading ref data from file: ', ref_filename_load)
        print("Reading reference video data from file: \n ", data_ref)

        data_ref, final_excluded_region_vid_ref1, df_ref_no_analysis = define_analysis_region("reference" , data_ref, vid_ref1_region, vid1_ref1_region_exclude)

        df_ref = min_max_normalized_dataFrame(data_ref, 
                                            cols_do =["frame_cnts", "frame_times", "frame_gray_av", "roi_surf_size", "roi_scinti_size"], 
                                            cols_not = ["x_com_LED", "y_com_LED"])
        add_scinti_state_to_df(df_ref, 0.4)
        df_ref = add_scinti_cycle_to_df(data_ref,max_gap_in_beam_ref_pts)


    #add stages to reference:
    data_ref_plateaus, data_ref_plateaus_stats = add_plateaus_to_reference(df_ref, length_refPlateaus, frame_rate, padding = frame_rate, reg=vid_ref1_region, exclude = final_excluded_region_vid_ref1)
    #from here, df_ref is shortened!

    print('Post-processed DataFrame from Video Analysis:\n', df_data_std)
    print('Post-processed DataFrame from Reference Video Analysis:\n', df_ref)

    #--------------------- post-processing of modified data frames / compute gating latencies-------------------------------------#
    lower_gating_levels = find_lower_gating_level_from_reference(data_ref_plateaus, data_ref_plateaus_stats)
    df_lower_gating_levels = pd.DataFrame.from_dict(lower_gating_levels, orient = 'index')
    print("Lower gating levels as df:", df_lower_gating_levels)

    if comp_lat:
        # compute gating latencies
        df_latencies, dict_latencies, data_mod= get_beamON_beamOFF_latencies(df_data_std, data_ref_plateaus, data_ref_plateaus_stats, lower_gating_levels, reg=vid1_region, reg_exclude = vid1_region_exclude)
        #latencies[method] = { "beamON lat"   : {"result": beamON_lat,  "uncert": beamON_lat_uncert},   "beamOFF lat"  : {"result": beamOFF_lat,  "uncert": beamOFF_lat_uncert},
        #                     "tON surf"      : {"result": tON_surf,    "uncert": tON_surf_uncert},     "tOFF surf"    : {"result": tOFF_surf,    "uncert": tOFF_surf_uncert},
        #                     "tON scinti"    : {"result": tON_scinti,  "uncert": tON_scinti_uncert},   "tOFF scinti"  : {"result": tOFF_scinti,  "uncert": tOFF_scinti_uncert}}
        #
        
        #total uncertainties latencies
        #pdb.set_trace()
        def total_results_to_dict(df_ltc):
            tot_res = {}
            for name, grp in df_ltc.groupby('method'):
                total_beamON_lat = grp['beamON_lat'].mean()
                total_beamON_lat_std = grp['beamON_lat'].std()
                total_beamON_lat_unc = np.sqrt(grp['beamON_lat_unc'].apply(lambda x: x*x).sum())/grp['beamON_lat_unc'].count()
                
                total_beamOFF_lat = grp['beamOFF_lat'].mean()
                total_beamOFF_lat_std = grp['beamOFF_lat'].std()
                total_beamOFF_lat_unc = np.sqrt(grp['beamOFF_lat_unc'].apply(lambda x: x*x).sum())/grp['beamOFF_lat_unc'].count()

                tot_res[name] = {'total_beamON_lat': total_beamON_lat, 'total_beamON_lat_std': total_beamON_lat_std, 'total_beamON_lat_unc':total_beamON_lat_unc, 
                                'total_beamOFF_lat': total_beamOFF_lat,'total_beamOFF_lat_std':total_beamOFF_lat_std,  'total_beamOFF_lat_unc':total_beamOFF_lat_unc }

                print(f'---- For method {name} ------')
                for key, value in tot_res[name].items():
                    print(key, value )
                print('-------------------------------')

            return tot_res

        total_results = total_results_to_dict(df_latencies)
        
        df_latencies_comparison_stats = add_statistics_for_development(df_latencies, cols_of_interest= ['beamON_lat', 'beamON_lat_unc', 'beamOFF_lat', 'beamOFF_lat_unc', 'tON_surf_unc', 'tOFF_surf_unc', 'duration surfON', 'duration surfON unc'])

        if save_postprocessed_results:
            data_mod.to_csv(filename_postprocessed_data,index=False)
            data_ref_plateaus.to_csv(filename_postprocessed_data_ref,index=False)
            print(f"\n DataFrames of post-processed data written to files!\n {filename_postprocessed_data} \n {filename_postprocessed_data_ref}")
            with open(filename_lower_gating_levels, 'wb') as f:
                pickle.dump(lower_gating_levels, f)
            print("Saved lower gating levels to file as dictionary!")

            df_data_std_no_analysis.to_csv(filename_postprocessed_excluded_data, index=False)
            df_ref_no_analysis.to_csv(filename_postprocessed_excluded_ref, index=False)

        if save_latency_results: 
            with open(filename_dict_latencies_save, 'wb') as fl:
                pickle.dump(dict_latencies, fl)
            df_latencies.to_csv(filename_latencies,index_label = 'cycle')
            print('Saved latencies to file:\n', filename_latencies )
            df_latencies_comparison_stats.to_csv(filename_latencies_stats, index_label = 'cycle')
            pd.DataFrame(total_results).to_csv(filename_total_results)
            print('Save total results to file: \n ', filename_total_results)

    else:
        data_mod=df_data

    print('\n Analysis completed.')

    data_ref_mod = data_ref_plateaus
### done analysis ############

print('I continue with loading the results from file for visualization...')
# Load Data...
data_mod = pd.read_csv(filename_postprocessed_data)
data_ref_mod = pd.read_csv(filename_postprocessed_data_ref)
with open(filename_dict_latencies_save, 'rb') as fload:
        dict_latencies = pickle.load(fload) 
df_latencies = pd.read_csv(filename_latencies,index_col=0)
df_latencies_comparison_stats = pd.read_csv(filename_latencies_stats, index_col=0)
total_results = pd.read_csv(filename_total_results, index_col=0)
with open(filename_lower_gating_levels, 'rb') as f_load:
    lower_gating_levels = pickle.load(f_load)

df_data_std_no_analysis = pd.read_csv(filename_postprocessed_excluded_data)
df_ref_no_analysis = pd.read_csv(filename_postprocessed_excluded_ref)

print("Latencies from file: \n", df_latencies)
print("Latencies comparison statistics from file: \n", df_latencies_comparison_stats)

######################## Output total results###################################################################
#total uncertainties:
beamON_tot_unc = np.sqrt(df_latencies['beamON_lat'].std()**2+df_latencies['beamON_lat_unc'].pow(2).sum())
beamOFF_tot_unc = np.sqrt(df_latencies['beamOFF_lat'].std()**2+df_latencies['beamOFF_lat_unc'].pow(2).sum())

print("------- total results with uncertainties---------------") 
print(f"beamON_lat = {round(df_latencies['beamON_lat'].mean(),1)} \pm {round(beamON_tot_unc,1)}")
print(f"beamOFFlat = {round(df_latencies['beamOFF_lat'].mean(),1)} \pm {round(beamOFF_tot_unc,1)}")

##########################################################################################################################
#############################                   PLOTTING                      ############################################
##########################################################################################################################
#TODO: move this to extra file...
print('\n I continue with Plotting the analysis results...')

surf_color = utils.h_colorcodes.lmu_orange#'darkorange'
surf_ref_color = utils.h_colorcodes.lmu_orange #'darkorange'
surf_LED_color = utils.h_colorcodes.lmu_rostrot_100 #'red'
surf_ref_LED_color = utils.h_colorcodes.lmu_rostrot_100 #'firebrick'

surf_exclude_color = utils.h_colorcodes.lmu_gray_50

scinti_color = utils.h_colorcodes.lmu_petrol_65#'lightskyblue'#(21/255, 0/255, 255/255) #utils.h_colorcodes.lmu_hellblau_65
scinti_ref_color = utils.h_colorcodes.lmu_petrol_65#'lightskyblue'#(21/255, 0/255, 255/255) #utils.h_colorcodes.lmu_hellblau_100

beamOFF_lat_color = utils.h_colorcodes.lmu_pink_100
beamOFF_lat_unc_color = utils.h_colorcodes.lmu_dunkelgruen_100

#plot frames being set above! plot_frame_col1,plot_frame_col2,plot_frame_col3

#TODO: export these demo figures to separate plotting-only file
#TODO: move all plotting functions to ONE file. of these the plotting-only file and this one can feed.
list_keys_dict_latencies = list(dict_latencies["surf"].keys())
min_lat_cycle = list_keys_dict_latencies[0]

'''figure for motivation gating latency
fig_demo = plt.figure()
axdemo = fig_demo.add_subplot(111)
axdemoB= axdemo.twinx()
axdemo.plot(data_mod["frame_times"]/1000 , data_mod["px_roi_surf"], lw = 2,color=surf_ref_color, label = 'surface')
axdemoB.plot(data_mod['frame_times']/1000, data_mod["scinti_on"],'-', markersize=1,color= scinti_ref_color, label = 'scintillator')
axdemo.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k', label = "lower gat. level (-) & uncert.(--)")
axdemo.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
axdemo.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
axdemo.add_patch(matplotlib.patches.Rectangle((dict_latencies["surf"][min_lat_cycle]["tON surf"]["result"]/1000, data_mod['px_roi_surf'].min() ), 
                                      (dict_latencies["surf"][min_lat_cycle]["tOFF surf"]["result"]/1000-dict_latencies["surf"][min_lat_cycle]["tON surf"]["result"]/1000), data_mod["px_roi_surf"].max()*1.01,
                                      facecolor = utils.h_colorcodes.lmu_dunkelgruen_100, alpha = 0.2, edgecolor=utils.h_colorcodes.lmu_dunkelgruen_100, linestyle = '-', angle=0.0, rotation_point='xy',
                                      label = 'surface in gating window'))
axdemo.add_patch(matplotlib.patches.Rectangle((dict_latencies["surf"][min_lat_cycle]["tON scinti"]["result"]/1000, data_mod["px_roi_surf"].min() ), 
                                      (dict_latencies["surf"][min_lat_cycle]["tOFF scinti"]["result"]/1000-dict_latencies["surf"][min_lat_cycle]["tON scinti"]["result"]/1000), data_mod["px_roi_surf"].max()*1.01, 
                                    facecolor = 'blue', alpha = 0.2, edgecolor= 'blue', linestyle = '--', angle=0.0, rotation_point='xy', label = 'irradiation'))

#plt.yticks([])
axdemo.set_xlabel("time [s]")
axdemo.set_xlabel('time [s]')
axdemo.set_ylabel('px cnts [1]')
axdemoB.yaxis.set_ticks([0, 1], ["off", "on"])
axdemoB.set_ylabel('scintillator binary [on/off]')
axdemo.set_xlim([3,28])
#plt.legend(loc='upper left')
# ask matplotlib for the plotted objects and their labels
lines, labels = axdemo.get_legend_handles_labels()
lines2, labels2 = axdemoB.get_legend_handles_labels()
axdemo.legend(lines + lines2, labels + labels2, loc='center left')

#plt.rc('font', size=15)          # controls default text 
plt.show()
end of demo figure '''

''' figure demo pure signal
figDEMOcurve = plt.figure()
axdem = figDEMOcurve.add_subplot(111)
axdemB = axdem.twinx()
axdemC = axdem.twinx()
axdem.plot(data_mod["frame_times"]/1000 , data_mod["px_roi_surf"],'.', lw = 3, color=surf_ref_color, label = 'surface')
axdemC.plot(data_mod["frame_times"]/1000, data_mod["px_roi_scinti"], '.', color="blue", label ="scintillator raw")
axdemB.plot(data_mod['frame_times']/1000, data_mod["scinti_on"],'-', markersize=1,color= utils.h_colorcodes.lmu_hellblau_100, label = 'scintillator')
#axdem.add_patch(matplotlib.patches.Rectangle((dict_latencies["surf"][min_lat_cycle]["tON scinti"]["result"]/1000, data_mod["px_roi_surf"].min() ), 
 #                                     (dict_latencies["surf"][min_lat_cycle]["tOFF scinti"]["result"]/1000-dict_latencies["surf"][min_lat_cycle]["tON scinti"]["result"]/1000), data_mod["px_roi_surf"].max()*1.01, 
  #                                  facecolor = 'blue', alpha = 0.2, edgecolor= 'blue', linestyle = '--', angle=0.0, rotation_point='xy', label = 'irradiation'))


# ask matplotlib for the plotted objects and their labels
lines, labels = axdem.get_legend_handles_labels()
lines2, labels2 = axdemB.get_legend_handles_labels()
lines3, labels3 = axdemC.get_legend_handles_labels()
axdem.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
axdem.set_xlim([404,465])
#axdem.set_ylim([1800,7100])
axdem.set_xlabel('time [s]')
axdem.set_ylabel('surf px cnts [1]')
axdemB.yaxis.set_ticks([0, 1], ["off", "on"])
#axdemB.set_ylabel('scintillator binary [on/off]')
axdemC.set_ylabel('scintillator px cnts [1]')
plt.show()

sys.exit()
#end demo''' 


''' fig demo reference 
figDEMOref = plt.figure()
axdemref = figDEMOref.add_subplot(111)
axdemref2 = axdemref.twinx()
axdemref.plot(data_ref_mod["frame_times"]/1000 , data_ref_mod["px_roi_surf"], '.', markersize=2,color=surf_ref_color, label = 'surface reference')
axdemref.axhline(y=lower_gating_levels["surf"]["total"], linestyle= '-',linewidth=1, color='k', label = "lower gat. level (-) & uncert.(--)")
axdemref.axhline(y=(lower_gating_levels["surf"]["total"]+lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
axdemref.axhline(y=(lower_gating_levels["surf"]["total"]-lower_gating_levels["surf"]["unc_total"]), linestyle= '--',linewidth=1, color='k')
axdemref2.plot(data_ref_mod["frame_times"]/1000 , data_ref_mod["scinti_on"],'-', markersize=1,color= scinti_ref_color, label = 'scintillator')
# ask matplotlib for the plotted objects and their labels
lines, labels = axdemref.get_legend_handles_labels()
lines2, labels2 = axdemref2.get_legend_handles_labels()
#axdemref.legend(lines + lines2, labels + labels2, loc='upper left')
#axdemref.set_xlim([96,118])
axdemref.set_xlim([23,207])
#axdemref.set_xlim([21,213])
#axdemref.set_ylim([6475,6700])
#axdemref.set_ylim([6100,7120])
axdemref.set_xlabel('time [s]')
axdemref.set_ylabel('px cnts [1]')
axdemref2.yaxis.set_ticks([0, 1], ["off", "on"])
axdemref2.set_ylabel('scintillator binary [on/off]')
plt.show()

end demo reference '''

''' figure demo setup, camera view
figDEMOcam = plt.figure()
# draw masked area on plotted frame
if not skip_video: 
    frame_list = [plot_frame_col1, plot_frame_col3]
    for frame in frame_list:
        cv2.rectangle(frame,(roi_surface[0],roi_surface[1]), (roi_surface[2],roi_surface[3]), (255,255,0), 2 ) # (img, left upper corner, right lower corner, color, thickness)
        cv2.rectangle(frame, (roi_scinti[0], roi_scinti[1]), (roi_scinti[2], roi_scinti[3]), (255,255,0),2)
plt.imshow(plot_frame_col1)
plt.show()'''

if show_final_plot:
    ffinal, ffinalframe = make_figures_ECMP2024()
    #ffinal.savefig("<insert-your-path-here-or-better-do-not-hard-set-it-but-use-the-dynamic-save-directory-in-configurations>/plot_DIBH.png", dpi=600)
    ffinal.show()
    #ffinalframe.savefig("<insert-your-path-here-or-better-do-not-hard-set-it-but-use-the-dynamic-save-directory-in-configurations>/frame_trio.png", dpi=600)
    if not skip_video:
        ffinalframe.show() #empty for test case

if report_PDF:
    #easy PDF report. #TODO: more sophisticated e.g. using FPDF and PDF class https://pyfpdf.readthedocs.io/en/latest/Tutorial/index.html#minimal-example or reportlab
    filename_pdfReport = os.path.join(save_directory_experiment_path, 'reportPDF')+'_'+exp_note + os.path.extsep + 'pdf'
    #figPDF = make_final_figure()
    figPDF, fig2 = make_figures_ECMP2024()
    title = figPDF.suptitle(f"Analysis of Gating Latencies \n {experiment}", 
                       backgroundcolor = "gainsboro")
    #title._bbox_patch._mutation_aspect = 0.04
    #title.get_bbox_patch().set_boxstyle("square", pad=11.9)
    figPDF.savefig(filename_pdfReport)
    #TODO: querformat?
    webbrowser.open_new(filename_pdfReport)
    
    latPDF = plt.figure(figsize=((11.6,8.2)))
    latPDF.suptitle(f"Analysis of Gating Latencies \n {experiment}", 
                       backgroundcolor = "gainsboro")
    latPDF.text(0.1, 0.8, df_latencies_comparison_stats.astype(float).round(1).to_string(), ha = "left", va = "top", fontsize = 10)
    latPDF.text(0.1, 0.4, (pd.DataFrame(total_results).T).to_string(), wrap=True) #TODO make this more pretty
    latPDF.text(0.1, 0.3, f"reported lower gating level: {round(lower_gating_levels['surf']['total'],2)} \pm {round(lower_gating_levels['surf']['unc_total'],2)}")

    filename_LATpdfReport = os.path.join(save_directory_experiment_path, 'reportPDF_onlyLatencies')+'_'+exp_note + os.path.extsep + 'pdf'
    latPDF.savefig(filename_LATpdfReport)
    webbrowser.open_new(filename_LATpdfReport)

print("Script completed!")

if keep_debugger_at_the_end:
    pdb.set_trace()
    #df_ground_truth = pd.read_csv(os.path.join(save_directory_experiment_path, "latencies_ground_truth_sineT15000.txt"))
    #df_comp_ground = df_ground_truth.merge(df_latencies, on='scinti_on_cycle')
    #df_comp_ground = df_comp_ground[["scinti_on_cycle", "true_beamON_lat", "true_beamOFF_lat", "beamOFF_lat", "beamON_lat"]]
    #df_comp_ground['true_beamON_lat-comp_beamON_lat']=df_comp_ground['true_beamON_lat']-df_comp_ground['beamON_lat']
    #df_comp_ground['true_beamOFF_lat-comp_beamOFF_lat']=df_comp_ground['true_beamOFF_lat']-df_comp_ground['beamOFF_lat']