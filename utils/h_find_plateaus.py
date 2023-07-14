from math import nan
from statistics import median, quantiles, stdev
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pdb
import sys

# Idea:
# 0. Find the ramp-up region
#    - segment data into: sine, DIBH, and ramp up: A, B, C
#    - Find the sine i.e. with a convoluion with sine
#    - 
#    -
# 1. roughly, find areas where a step ist to be expected: B-> bi
#    - rolling mean over data with window size = second or half plateau length?
#    - compute the variance or diff of this rolled data
#    - identify the regions of interest
# 2. exactly, identify the position of a step
#    - iterate over those roughly determined regions of interest
#    - convolve the region of interest's original data points with a step function
#    - identify the begin of a step
# 3. write to df:
#    - name the steps (i.e. 1, 2, 3, ... ), find their mean value
#    - write to data frame in a way that in the original data each row gets a step and mean written
#    - pass this on to find lower gating level function

###########################################

def analyze_variances_in_rampup(df_ramp, window_size, fpsloc,list_of_excluded_indeces):
    '''
    window_size in numbers data pts, good choice: full plateaus length, i.e. if 10 seconds plateaus-> w=10*FPS
    rolling variance: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
    - center = True: set the window labels as the center of the window index.
    - at the edges: 
    '''
    # rolling the data and create variance
    print('window_size to analyze variance: ', window_size)
    rolling_variance = df_ramp['px_roi_surf'].rolling(int(window_size), center=True).var()
    #--> results of windows in including a nan, gives nan, which will not exceed the below threshold.
    df_ramp_mod = df_ramp.assign(px_roi_surf_rolledVar=rolling_variance)
    
    #insert nans in ecluded region 
    # #TODO: this is inefficient (took 30s for 9k pts) see documentation in my notion. is there another ways?
    # # it would be faster if I would drop the exclude values in the next function (find_candidates) and deal with the change in size of the array
    for ix in list_of_excluded_indeces: 
        df_ramp_mod.loc[ix, "px_roi_surf_rolledVar"] = nan
    df_ramp_mod = df_ramp_mod.sort_index()
    # this will change df_ramp_mod's shape! and wiil change df_ramp_mods change! -> remove them again
    
    #regions of interest for rough analysis: when exceeding variance threshold
    threshold = df_ramp_mod['px_roi_surf_rolledVar'].quantile(0.25) # define threshold
    df_ramp_mod = df_ramp_mod.assign(varLabels=lambda x: np.where(x['px_roi_surf_rolledVar']>threshold, True, False))

    '''
    #remove rows that were excluded originally:
    df_ramp_mod = df_ramp_mod.drop(list_of_excluded_indeces)
    '''
    return df_ramp_mod

def sort_candidates_into_dict_and_add_cand_to_df(df_ramp):
    #TODO: get rid of the dict output (und use a groupby approach in find_step_positions)
    #TODO: this one is quite slow (I am iterating over the index)
    cand_cnt=1
    cand_dict={}
    df_ramp_mod1 = df_ramp.copy(deep=True)
    idx=df_ramp_mod1.index.min()

    while idx in range(df_ramp_mod1.index.max()):# ugly but in my case sufficient way 
        #print(idx)
        if df_ramp_mod1.at[idx, 'varLabels']:
            cand_df = pd.DataFrame(columns=df_ramp_mod1.columns)
            cand_df = pd.concat([cand_df, pd.DataFrame([df_ramp_mod1.loc[idx].to_dict()])])
            df_ramp_mod1.loc[idx,'step_cand_group'] = f'cand{cand_cnt}'
            #print(idx)
            idx+=1
            while df_ramp_mod1.at[idx, 'varLabels']:
                cand_df = pd.concat([cand_df, pd.DataFrame([df_ramp_mod1.loc[idx].to_dict()])])
                df_ramp_mod1.loc[idx,'step_cand_group'] = f'cand{cand_cnt}'
                idx+=1
            #only keep those > 120 dt.pts:
            #TODO: exclude by statistics not random number
            if cand_df.shape[0] > 120:
                cand_dict[f'cand{cand_cnt}'] = cand_df
                cand_cnt+=1
            else: #remove cand description again
                #pdb.set_trace()
                df_ramp_mod1.loc[df_ramp_mod1["step_cand_group"]==f'cand{cand_cnt}', 'step_cand_group']=nan
          
        else:
            idx+=1
    return cand_dict, df_ramp_mod1

def get_candidates_regions_add_to_df(df_ramp_mod, list_of_excluded_indeces):
    df_copy = df_ramp_mod.copy(deep=True)

    indx_changes_cands = df_copy.ne(df_copy.shift()).filter(like='varLabels').apply(lambda x: x.index[x].tolist())['varLabels'].to_list()
    #ignore the first:
    indx_changes = indx_changes_cands[1:]
    #drop those within exclude, to be on the safe side (they should not be there)
    indx_changes = [x for x in indx_changes if x not in list_of_excluded_indeces]
    print("indx_changes in reference: ", indx_changes)

    cand_list = []
    cand_cnt = 0
    
    for pos, idx in enumerate(indx_changes):
        #ending of True-region
        if df_copy.at[idx,'varLabels'] ==False and df_copy.at[idx-1,'varLabels']==True:
            cand_cnt +=1
            cand_name_prev = f'cand{cand_cnt}'
        #ending of false-region
        else: #==True:
            cand_name_prev = nan

        if idx == np.min(indx_changes):
            factor = indx_changes[pos]-df_copy.index.min()#=idx
        else: 
            factor = indx_changes[pos]-indx_changes[pos-1]
        
        if factor > 120:
            cand_list += factor *[cand_name_prev]
            #print(idx)
        else:
            if df_copy.at[idx,'varLabels'] ==False and df_copy.at[idx-1,'varLabels']==True:
                cand_cnt -= 1
            cand_list += factor *[nan]

        #test
        if len(cand_list) != idx-df_copy.index.min():
            print('there is sth to check here')
            pdb.set_trace()

    #and the latter area from last idx in indx_changes until end of dataframe
    if df_copy.at[idx, 'varLabels'] == True and df_copy.at[idx-1,'varLabels']==False:
        last_cand_entry = f'cand{cand_cnt}'
        cand_cnt +=1
    else:
        last_cand_entry = nan

    last_factor = df_copy.index.max()-idx+1
    if last_factor <120:
        last_cand_entry = nan
        cand_cnt -=1  

    cand_list += last_factor *[last_cand_entry]
    
    #print("FULL len(cand_list): ", len(cand_list), ", FULL len df_copy: ", df_copy.shape[0])

    #test
    #TODO: this holds only for df without exclude regions
    if len(cand_list) != df_copy.index.max()-(df_copy.index.min()-1): 
        print('there is sth to check here')
        pdb.set_trace()

    df_ramp_new = df_copy.assign(step_cand_group= cand_list)
    
    return df_ramp_new

def find_jumps_positions(df):
    
    df['roi_ampl_step'] = False
    df['step_indx'] = False

    for candname, cand_group in df.groupby('step_cand_group'):
        df_only_candX = cand_group
        
        roi_ampl = cand_group['px_roi_surf'].copy(deep=True)
        roi_ampl -= np.average(roi_ampl)
        step_fun = np.hstack((np.ones(len(roi_ampl)), -1*np.ones(len(roi_ampl))))
        roi_ampl_step = np.convolve(roi_ampl, step_fun, mode='valid')

        # index of peak of convolution (indx within roi_ampl_step)
        step_indx = np.argmax(roi_ampl_step)

    	#TODO move this into one plot below
        cand_group.loc[:,'roi_ampl_step']=roi_ampl_step[:-1]
        cand_group.loc[cand_group.index.min()+step_indx, 'step_indx'] = True
        df.loc[cand_group.index.min()+step_indx, 'step_indx'] = True

        ''' all candidates plots'''
        '''
        f2 = plt.figure()
        plt.plot(range(len(roi_ampl)),roi_ampl,'.', label='orig')
        plt.plot(range(len(roi_ampl_step)),roi_ampl_step/50, label='step func /50')
        plt.plot((step_indx, step_indx), (roi_ampl_step[step_indx]/50, 0), 'r', label=f"mid at {step_indx}")
        plt.title(candname)
        f2.show()
        '''

    return df


def find_steps_positions(dict, df_ramp_mod):
    '''
    for each step candidate, convolve its amplitude with a matching step function,
    ---
    np.convolve(..., ‘valid’) :
    Mode ‘valid’ returns output of length max(M, N) - min(M, N) + 1. 
    The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect.
    '''
    # pseudo: 
    # - for step_cand in cand_dict, find the exact step in frame
    # - (round it to frame if needed)
    
    df_ramp_mod['roi_ampl_step'] = False
    df_ramp_mod['step_indx'] = False

    for step_cand in dict:
        df_only_candX = df_ramp_mod[df_ramp_mod['step_cand_group']==step_cand]
        
        roi_ampl = df_only_candX['px_roi_surf'].copy(deep=True)
        roi_ampl -= np.average(roi_ampl)
        step_fun = np.hstack((np.ones(len(roi_ampl)), -1*np.ones(len(roi_ampl))))
        roi_ampl_step = np.convolve(roi_ampl, step_fun, mode='valid')

        # index of peak of convolution (indx within roi_ampl_step)
        step_indx = np.argmax(roi_ampl_step)

    	#TODO move this into one plot below
        df_only_candX.loc[:,'roi_ampl_step']=roi_ampl_step[:-1]
        df_only_candX.loc[df_only_candX.index.min()+step_indx, 'step_indx'] = True
        df_ramp_mod.loc[df_only_candX.index.min()+step_indx, 'step_indx'] = True


 
        ''' all candidates plots'''
        '''
        f2 = plt.figure()
        plt.plot(range(len(roi_ampl)),roi_ampl,'.', label='orig')
        plt.plot(range(len(roi_ampl_step)),roi_ampl_step/50, label='step func /50')
        plt.plot((step_indx, step_indx), (roi_ampl_step[step_indx]/50, 0), 'r', label=f"mid at {step_indx}")
        plt.title(step_cand)
        f2.show()
        '''
    
'''
def match_frames_to_step_and_mean(df_ramp_mod, padding=120):
    # walk over df_ramp_mod and add plateau names per entry, then groupby them, then return a dict of plateaus (from where to where with mean)
    dict_of_plateaus ={}
    df_ramp_mod['plateau']=np.nan
    plateau_cnt=0
    idx=df_ramp_mod.index.min()

    while idx in range(df_ramp_mod.index.max()):
        #print(idx)
        if df_ramp_mod.at[idx,'step_indx']==False:
            idx+=1
        elif df_ramp_mod.at[idx,'step_indx']:
            plateau_cnt+=1
            df_ramp_mod.loc[idx,'plateau']=plateau_cnt
            idx+=1
            while idx in range(df_ramp_mod.index.max()) and not df_ramp_mod.at[idx,'step_indx']:
                df_ramp_mod.loc[idx,'plateau']=plateau_cnt
                idx+=1
            local_df = df_ramp_mod[df_ramp_mod['plateau']==plateau_cnt]
            start_frame = local_df.at[local_df.index.min(),'frame_cnts']
            start_time = local_df.at[local_df.index.min(), 'frame_times']
            end_frame = local_df.at[local_df.index.max(),'frame_cnts']
            end_time = local_df.at[local_df.index.max(), 'frame_times']
            
            mean = local_df.loc[(local_df.index.min()+int(padding/2)):(local_df.index.max()-int(padding/2)),'px_roi_surf'].mean()
            std_dev = local_df.loc[(local_df.index.min()+int(padding/2)):(local_df.index.max()-int(padding/2)),'px_roi_surf'].std()
            median = local_df.loc[(local_df.index.min()+int(padding/2)):(local_df.index.max()-int(padding/2)),'px_roi_surf'].median()
            percentile_2p5 = local_df.loc[(local_df.index.min()+int(padding/2)):(local_df.index.max()-int(padding/2)),'px_roi_surf'].quantile(0.025)
            percentile_97p5= local_df.loc[(local_df.index.min()+int(padding/2)):(local_df.index.max()-int(padding/2)),'px_roi_surf'].quantile(0.975)
            #get start and end frame via grouping
            dict_of_plateaus[plateau_cnt]= {'start_frame': start_frame, 'end_frame': end_frame, 'start_time': start_time, 'end_time': end_time, 'mean': mean, 'std': std_dev, 'median':median, 'percentile_2p5':percentile_2p5, 'percentile_97p5': percentile_97p5 }
            #pdb.set_trace()
        else:
            sys.exit('error in match frames')
    for plateau,d in dict_of_plateaus.items():
        plt.plot((d['start_time']/1000, d['end_time']/1000), (d['median'], d['median']), 'orange')
        plt.fill_between((d['start_time']/1000, d['end_time']/1000), (d['percentile_2p5'], d['percentile_2p5']), (d['percentile_97p5'], d['percentile_97p5']), step='post', color='tab:blue', alpha=0.2)
    plt.show()

    #add mean/std plateaus to DataFrame:
    df_ramp_mod.loc[:,'mean plateau'] = 0
    df_ramp_mod.loc[:,'std plateau'] = 0
    for plateaux,dx in dict_of_plateaus.items():
        #pdb.set_trace()
        #loc_df = df_rampup_mod[df_rampup_mod['plateau']==plateaux]
        #loc_df.loc[:,'mean plateau']=dx['mean']
        #loc_df.loc[:,'std plateau'] = dx['std']
        df_ramp_mod.loc[df_ramp_mod.plateau==plateaux,'mean plateau']=dx['mean']
        df_ramp_mod.loc[df_ramp_mod.plateau==plateaux,'std plateau']=dx['std']

    return dict_of_plateaus #dict = {'1': {'start_frame': ...., 'end frame' : ..., 'mean': ...}, '2': ....}
'''

def match_frames_to_plateaus(df_ramp_mod):
    #TODO: das könnte ich auch noch schneller machen mit indx_changes statt iteration -> siehe get_candidates regions
    df_ramp_mod['plateau']=np.nan
    
    plateau_cnt=0
    idx=df_ramp_mod.index.min()

    while idx in range(df_ramp_mod.index.max()):
        #print(idx)
        if df_ramp_mod.at[idx,'step_indx']==False:
            idx+=1
        elif df_ramp_mod.at[idx,'step_indx']:
            plateau_cnt+=1
            df_ramp_mod.loc[idx,'plateau']=plateau_cnt
            idx+=1
            while idx in range(df_ramp_mod.index.max()) and not df_ramp_mod.at[idx,'step_indx']:
                df_ramp_mod.loc[idx,'plateau']=plateau_cnt
                idx+=1
        else:
            sys.exit('error in match frames')
    return df_ramp_mod


def match_stats_to_frames_and_plateaus(df_ramp_mod, padding=120):
    ''' only use for plotting, otherwise use stats_frame'''

    for plateau_name, group in df_ramp_mod.groupby('plateau'):
       
        min_index_this_group = df_ramp_mod[df_ramp_mod['plateau']==plateau_name].index.min()
        max_index_this_group = df_ramp_mod[df_ramp_mod['plateau']==plateau_name].index.max()
     
        df_ramp_mod.loc[(min_index_this_group+padding):(max_index_this_group-padding),'plateauWpad'] = plateau_name
    
    for padded_plateau, grp in df_ramp_mod.groupby('plateauWpad'):
        df_ramp_mod.loc[grp.index.min():grp.index.max(),'median'] = grp['px_roi_surf'].median()
        df_ramp_mod.loc[grp.index.min():grp.index.max(),'mean'] = grp['px_roi_surf'].mean()
        df_ramp_mod.loc[grp.index.min():grp.index.max(),'std'] = grp['px_roi_surf'].std()
        df_ramp_mod.loc[grp.index.min():grp.index.max(),'percentile_2p5'] = grp['px_roi_surf'].quantile(0.025)
        df_ramp_mod.loc[grp.index.min():grp.index.max(),'percentile_97p5'] = grp['px_roi_surf'].quantile(0.975)

    return df_ramp_mod