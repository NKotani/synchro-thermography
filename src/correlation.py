import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib.ticker as ticker

import utils

def correlation(heartbeats, temperatures, savename, mean_rr_frame, savename_append=0, FPS=30, isRaw=False, isSave=True):
    utils.set_plt()
    lags, xc_mp, fig, ax  = plt.xcorr(heartbeats, temperatures, detrend=mlab.detrend_linear, maxlags=mean_rr_frame)
    plt.xlabel(f'Lag [frame] (FPS={FPS})')
    plt.ylabel('Correlation Coefficient')
    if isRaw:
        savename_append = f'{savename_append}Raw'
    if isSave:
        plt.savefig(savename+f'correlogram{savename_append}.png', dpi=150, bbox_inches='tight')
    plt.close()
    max_index = np.argmax(np.abs(xc_mp))
    if lags[max_index] >= 1 and isSave:
        print(f'Maximum correlation coefficient: {utils.np_round(xc_mp[max_index],2)} ({xc_mp[max_index]:.5f})')
    elif isSave:
        print(f'Maximum correlation coefficient: {utils.np_round(xc_mp[max_index],2)} ({xc_mp[max_index]:.5f})')
    return xc_mp[max_index]

def create_wave(len_temperatures, mean_rr_frame, savename, FPS=30, isRaw=False):
    t = np.arange(0, len_temperatures)
    wave = np.cos(2 * np.pi * t / mean_rr_frame)
    utils.set_plt()
    plt.plot(t/FPS, wave)
    plt.xlim(0, len_temperatures/FPS)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    savename = savename+'wave.png'
    if isRaw:
        savename = savename+'waveRaw.png'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.close()
    return wave

def plt_correlation(image, list_of_coordinates, correlation_list, savename):
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(image)
    x = np.array(list_of_coordinates)[:,0]
    y = np.array(list_of_coordinates)[:,1]
    alpha_values = np.clip(np.abs(correlation_list) / 0.7, 0, 1)
    ax.scatter(y, x, label='Correlation between temperature and heart rate', color=utils.COLORLIST_PLT_A11Y[5], alpha=alpha_values, s=2)
    plt.savefig(savename[:-17] + 'correlation.png')
    plt.close()
    return

def run(heartbeats, temperatures, mean_rr_frame, savename, list_of_coordinates=[[175, 350], [275, 350]], isRaw=False, FPS=30, area_index=2):
    wave = create_wave(temperatures.shape[0], mean_rr_frame, savename, FPS, isRaw)
    for i, coordinate in enumerate(list_of_coordinates):
        x = coordinate[0]
        y = coordinate[1]
        mean_temperature = np.mean(temperatures[:,x-area_index:x+area_index+1,y-area_index:y+area_index+1], axis=(1,2))
        correlation(wave, mean_temperature, savename[:-7], mean_rr_frame, i, FPS, isRaw)
    return

def search_rectangle(temperatures, mean_rr_frame, savename, list_of_coordinates=[[175, 350], [275, 350]], image=np.zeros(1), isRaw=False, FPS=30, area_index=2):
    correlation_list = []
    wave = create_wave(temperatures.shape[0], mean_rr_frame, savename, FPS, isRaw)
    for i, coordinate in enumerate(list_of_coordinates):
        x = coordinate[0]
        y = coordinate[1]
        mean_temperature = np.mean(temperatures[:,x-area_index:x+area_index+1,y-area_index:y+area_index+1], axis=(1,2))
        correlation_list.append(correlation(wave, mean_temperature, savename[:-7], mean_rr_frame, i, FPS, isRaw, False))
    correlation_list = np.array(correlation_list)
    top_indices = np.argsort(np.abs(correlation_list))[-10:][::-1]
    print("Top 10 correlation indices:", top_indices)
    top_coordinates=[]
    for idx in top_indices:
        print(f"Coordinate: {list_of_coordinates[idx]}, Correlation: {correlation_list[idx]}")
        top_coordinates.append(list_of_coordinates[idx])
    print(f"Top 10 correlation mean value: {utils.np_round(np.mean(np.abs(correlation_list[top_indices])), 2)}")
    plt_correlation(image, list_of_coordinates, correlation_list, savename)
    return top_coordinates