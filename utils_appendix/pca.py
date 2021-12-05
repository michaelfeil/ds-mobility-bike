"""
author: Alvaro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib.gridspec as gridspec
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()


def get_colors(inp, colormap, vmin=None, vmax=None):
    """
    Convert from an array of numbers RGBA code colors
    """
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def correlationMatrix(data: pd.DataFrame):
    """
    Compute the correlation matrix given the dataframe
    
    returns: 
        Dataframe to use for PCA
    """
    cols_to_drop = ['feat_pickupgender_unk', 'feat_pickupgender_m',
           'feat_pickupgender_f', 'feat_pickupmeanbyear', 'feat_meantriptime',
           'feat_meandistancetrip', 'ana_dayoftheyear', 'ana_minuteofday']

    df = data.drop(columns=cols_to_drop, axis=1)

    # Change dtype of boolean cols related to weather
    weather_cols = [col for col in df.columns if 'w_c' in col]
    df[weather_cols] = df[weather_cols].astype(int)

    # We need encode 'feat_dayoftheweek' and 'feat_isholiday'
    df['day_of_week_sin'] = np.sin(df['feat_dayoftheweek'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['feat_dayoftheweek'] * (2 * np.pi / 7))
    df['feat_isholiday'] = pd.get_dummies(df['feat_isholiday'], drop_first=True, prefix='Holiday')

    df.drop(columns='feat_dayoftheweek', inplace=True)

    # Make sure we do not have booleans, since that will lead to missunderstanding to the model
    assert any(df.dtypes != bool)


    # Compute the correlation matrix
    """
    Since the formula for calculating the correlation coefficient standardizes the variables, changes 
    in scale or units of measurement will not affect its value. For this reason, normalizing will NOT 
    affect the correlation.
    """
    corr = df.corr()   

    # Generate a mask for the upper diagonal of the correalation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(50, 60))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                     annot=True, annot_kws={'size': 25})

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25, rotation=75)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 25, rotation=0)
    
    return df


def PCAnalysis(data: pd.DataFrame):
    """
    PCA analysis
    
    Attributes:
        - target: pickups_per_h
    
    return::None
        Plots
    """
    df = data.copy(deep=True)
    target = 'pickups_per_h'

    # Before doing PCA we have to center and scale the data. We don't need to standarize the cos or sin
    not_scaled = [col for col in df.columns if re.search("cos|sin|_w_c_", col)] + ['feat_isholiday', target]

    array_std = StandardScaler().fit_transform(df.drop(columns=not_scaled, axis=1)) 

    # Convert into a df and add the rest
    df_std = pd.DataFrame(data=array_std, columns=[i for i in df.columns if i not in not_scaled], index=df.index)
    df_std = pd.concat([df_std, df[not_scaled]], axis=1)

    # --------------------------------PCA-----------------------------------------
    X_std = df_std.drop(columns=target, axis=1)
    y = df_std[target].reset_index(drop=True)

    # Instantiate a PCA object. Let's say we want to reach 85% of variation
    pca = PCA(.85)

    # Calculate the loading scores and the variation each PC accounts for and generate coordinates for PCA graph 
    # based on the loading scores
    scores = pca.fit_transform(X_std.values)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)  # percentage in variation
    labels = [f'PC{i}' for i in range(1, len(per_var)+1)]

    pca_scores = pd.DataFrame(scores, columns=labels)
    final_pca_df = pd.concat([pca_scores, y], axis=1)

    # Apply mask for different classes p<2000 (0) | 2000<p<5000 (1) | p>5000 (3) WE CAN CHANGE THIS TO MORE CLASSES    
    y[y < 1000] = 0
    y.loc[y[y >= 1000].index.intersection(y[y < 2000].index)] = 1
    y.loc[y[y >= 2000].index.intersection(y[y < 3000].index)] = 2
    y.loc[y[y >= 3000].index.intersection(y[y < 4000].index)] = 3
    y.loc[y[y >= 4000].index.intersection(y[y < 5000].index)] = 4
    y.loc[y[y >= 5000].index.intersection(y[y < 6000].index)] = 5
    y[y >= 6000] = 6
    final_pca_df['pickups_per_h'] = y
    
    # We will use it this dict to plot a legend
    group_class = {0:'picks < 1000', 
                   1:'1000 <= picks < 2000', 
                   2:'2000 <= picks < 3000', 
                   3:'3000 <= picks < 4000',
                   4:'4000 <= picks < 5000',
                   5:'5000 <= picks < 6000',
                   6:'picks >= 6000'}

    final_pca_df.set_index(df_std.index, inplace=True)

    # loading scores
    pca_loadings = pd.DataFrame(pca.components_.T, columns=labels, index=X_std.columns)

    # -----------------------------PLOTS-------------------------------------
    fig = plt.figure(figsize=(30,50))
    G = gridspec.GridSpec(16, 8)

    # Scree plot
    ax1 = fig.add_subplot(G[:4, :])
    ax1.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    ax1.set_ylabel('Percentage of Explained Variance', fontsize=25)
    ax1.set_xlabel('Principal Components', fontsize=25)
    ax1.tick_params(labelsize=25)
    ax1.set_title('$\mathbf{Plot \; 1}$ - Scree Plot', fontsize=25)

    # Plot PC1 vs PC2
    # Plot circle - Create a list of 500 points with equal spacing between -1 and 1
    ax2 = fig.add_subplot(G[4:12, :])
    ax2.set_xlabel(f'PC1 - {per_var[0]}%', fontsize=25)
    ax2.set_ylabel(f'PC2 - {per_var[1]}%', fontsize=25)
    ax2.tick_params(labelsize=25)
    ax2.set_title('$\mathbf{Plot \; 2}$ - PC1 vs PC2', fontsize=25)
    
    x = np.linspace(start=-1,stop=1,num=500)
    # Find y1 and y2 for these points
    y_positive = lambda x: np.sqrt(1-x**2) 
    y_negative = lambda x: -np.sqrt(1-x**2)
    ax2.plot(x, list(map(y_positive, x)), color='maroon')
    ax2.plot(x, list(map(y_negative, x)),color='maroon')

    # Plot smaller circle
    x = np.linspace(start=-0.5,stop=0.5,num=500)
    y_positive = lambda x: np.sqrt(0.5**2-x**2) 
    y_negative = lambda x: -np.sqrt(0.5**2-x**2)
    ax2.plot(x, list(map(y_positive, x)), color='maroon')
    ax2.plot(x, list(map(y_negative, x)),color='maroon')

    # Create broken lines
    x = np.linspace(start=-1, stop=1, num=30)
    ax2.scatter(x, [0]*len(x), marker='_', color='maroon')
    ax2.scatter([0]*len(x), x, marker='|', color='maroon')

    add_string = ""
    for i in range(len(pca_loadings)):
        xi = pca_loadings.iloc[i, 0]
        yi = pca_loadings.iloc[i, 1]
        ax2.arrow(0,0, 
                  dx=xi, dy=yi, 
                  head_width=0.03, head_length=0.03, length_includes_head=True)
        
        add_string=f" ({round(xi,2)} {round(yi,2)})"
        ax2.text(pca_loadings.iloc[i, 0], 
                 pca_loadings.iloc[i, 1], 
                 s=pca_loadings.index[i] + add_string, fontsize=25)

    # Assign colors to the array of classes
    colors = get_colors(final_pca_df['pickups_per_h'].unique(), plt.cm.jet)[:, :3] # return a rgba. We donøt care about the alpha
    
    # cdict = {0: 'red', 1: 'yellow', 2: 'green'} # We can change this to more classes
    cdict = {i:j for i,j in enumerate(colors)}

    ax3 = fig.add_subplot(G[12:16, :])
    for g in final_pca_df['pickups_per_h'].unique():
        ix = np.where(final_pca_df['pickups_per_h'] == g)
        ax3.scatter(final_pca_df.iloc[ix]['PC1'], final_pca_df.iloc[ix]['PC2'], c = cdict[g], label = (g, group_class[g]), s = 50)
        ax3.legend()
        ax3.set_xlabel('PC1', fontsize=25)
        ax3.set_ylabel('PC2', fontsize=25)
        ax3.tick_params(labelsize=25)
        ax3.set_title('$\mathbf{Plot \; 3}$ - Number of pickups_per_h on PC1 and PC2', fontsize=25)


#     # Create 3D figures with different angles of visualization
#     ax4 = fig.add_subplot(G[16:20, :4], projection='3d')
#     for g in final_pca_df['pickups_per_h'].unique():
#         ix = np.where(final_pca_df['pickups_per_h'] == g)
#         ax4.scatter3D(final_pca_df.iloc[ix]['PC1'], final_pca_df.iloc[ix]['PC2'], final_pca_df.iloc[ix]['PC3'], color = cdict[g], label = g, s = 50)
#         ax4.set_xlabel('PC1')
#         ax4.set_ylabel('PC2')
#         ax4.set_zlabel('PC3')
#         ax4.set_title('$\mathbf{Plot \; 4}$ - Number of pickups_per_h on PC1, PC2 and PC3')
#         ax4.legend()
#         ax4.view_init(elev=45, azim=90) # angle of elevation and angle of azimut (in degrees)
        
#     ax5 = fig.add_subplot(G[16:20, 4:], projection='3d')
#     for g in final_pca_df['pickups_per_h'].unique():
#         ix = np.where(final_pca_df['pickups_per_h'] == g)
#         ax5.scatter3D(final_pca_df.iloc[ix]['PC1'], final_pca_df.iloc[ix]['PC2'], final_pca_df.iloc[ix]['PC3'], c = cdict[g], label = g, s = 50)
#         ax5.set_xlabel('PC1')
#         ax5.set_ylabel('PC2')
#         ax5.set_zlabel('PC3')
#         ax5.set_title('$\mathbf{Plot \; 5}$ - Like Plot 4 with different point of view')
#         ax5.legend()
#         ax5.view_init(elev=135, azim=90)
        
    # Add some padding
    fig.subplots_adjust(hspace=10, wspace=5)
    
    return final_pca_df, pca_loadings, X_std