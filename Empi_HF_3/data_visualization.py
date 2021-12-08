"""Cikk ábráinak elkészítése"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def figure3(nodes,G):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 3',size=25)

    """plot1"""
    """Plot the average of Degree Centrality across age and gender"""
    nodes_w_centrality = nodes.set_index("user_id").merge(
        pd.Series(dict(nx.degree_centrality(G))).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    
    nodes_w_centrality = nodes_w_centrality.rename({0: "Centrality"}, axis=1)
    plot_df = (
        nodes_w_centrality.groupby(["AGE", "gender"]).agg({"Centrality": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(ax=axes[0],data=plot_df, x="AGE", y="Centrality", hue="gender").set_ylabel("Degree Centrality")

    """plot2"""
    """Plot the average of Neighbor Connectivity across age and gender"""
    nodes_w_connectivity = nodes.set_index("user_id").merge(
        pd.Series(dict(nx.average_neighbor_degree(G))).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )

    nodes_w_connectivity = nodes_w_connectivity.rename({0: "Connectivity"}, axis=1)
    plot_df = (
        nodes_w_connectivity.groupby(["AGE", "gender"]).agg({"Connectivity": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(ax=axes[1],data=plot_df, x="AGE", y="Connectivity", hue="gender").set_ylabel("Neighbor Connectivity")

    
    """plot3"""
    """Plot the average of clustering coefficient across age and gender"""
    nodes_w_cc = nodes.set_index("user_id").merge(
        pd.Series(dict(nx.clustering(G))).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
  
    nodes_w_cc = nodes_w_cc.rename({0: "CC"}, axis=1)
    plot_df = (
        nodes_w_cc.groupby(["AGE", "gender"]).agg({"CC": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(ax=axes[2],data=plot_df, x="AGE", y="CC", hue="gender").set_ylabel("Clustering Coefficient")
    
def figure4(edges_w_features):
    
    """Férfiak életkorát az x embernél negatívra állítja (mint a cikkben)"""
    edges_w_features["AGE_x"] = np.where(edges_w_features["gender_x"]==1, -edges_w_features["AGE_x"], edges_w_features["AGE_x"]) 
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Figure 4', size=25)

    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )

    #Plot1
    """Nők ismerőseinek kor és nem szerinti eloszlása a nők kora szerint"""
    """nő-nő"""
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    
    """nő-fiú"""
    plot_df_w_m = plot_df.loc[(1, 0)].reset_index()
    
    """ebben már mindkét előző benne lesz"""
    plot_df_w=plot_df_w_w.append(plot_df_w_m) 
    plot_df_heatmap = plot_df_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    
    """Az oszlopok összegével osztjuk mindenhol, így mindenhol relatív gyakoriságok lesznek"""
    plot_df_heatmap_logged = plot_df_heatmap.div(plot_df_heatmap.sum(axis=1), axis=0)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[0])
    axes[0].set_xlabel("Age (Female)")
    axes[0].set_ylabel("Demographic distribution of friends")

    #Plot2
    """Ugyanaz a logika mint előbb, csak férfiakra"""
    plot_df_m_m = plot_df.loc[(1, 1)].reset_index()
    plot_df_m_w = plot_df.loc[(0, 1)].reset_index()
    plot_df_m=plot_df_m_m.append(plot_df_m_w)
    plot_df_heatmap = plot_df_m.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    
    plot_df_heatmap_logged = plot_df_heatmap.div(plot_df_heatmap.sum(axis=1), axis=0)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[1])
    axes[1].set_xlabel("Age (Male)")
    axes[1].set_ylabel("Demographic distribution of friends")
    

def figure5(edges_w_features):
    """Heatmap-ek gender páronként, valamint összesítve, kor szerint (logaritmizált skálán) """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Figure 5',size=25)

    #Plot1 - All-all
    #Itt nem-től függetlenül kell a csoportosítás kor alapján
    plot_df = edges_w_features.groupby(["AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    plot_df_heatmap, plot_df
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[0,0])
    axes[0,0].set_xlabel("Age")
    axes[0,0].set_ylabel("Age")

    #Plot2 - Men - Men
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    #Itt csak a férfi-férfi párok kellenek
    plot_df_m_m = plot_df.loc[(1, 1)].reset_index()
    plot_df_heatmap = plot_df_m_m.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[0,1])
    axes[0,1].set_xlabel("Age (Male)")
    axes[0,1].set_ylabel("Age (Male)")

    #Plot3 - Women - Women
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    #itt csak a nő-nő párok kellenek
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[1,0])
    axes[1,0].set_xlabel("Age (Female)")
    axes[1,0].set_ylabel("Age (Female)")

    #Plot4 - Men - Women
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    #Itt kell a fiú-lány és a lány-fiú kombináció is
    plot_df_w_m = plot_df.loc[(0, 1)].reset_index() #x a lány
    plot_df_m_w = plot_df.loc[(1, 0)].reset_index() #x a fiú
    plot_df_m_w.rename(columns={"AGE_x":"AGE_y","AGE_y":"AGE_x"},inplace=True) #a másodikban kicserélem az oszlopokat így ott is x a lány

    plot_df_all=plot_df_w_m.append(plot_df_m_w) #egymás alá rakja a kettőt, itt is x a lány

    plot_df_final = plot_df_all.groupby(["AGE_x", "AGE_y"]).agg(
        {"smaller_id": "sum"}
    ) #Mivel az egymás alá rakás miatt az x esetében minden age 2-szer szerepel, ezért Age_x szerint csoportosítom és összeadom a két darabszámot

    plot_df_heatmap = plot_df_final.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0).sort_values("AGE_x",ascending=False)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged,ax=axes[1,1])
    axes[1,1].set_xlabel("Age (Male)")
    axes[1,1].set_ylabel("Age (Female)")
