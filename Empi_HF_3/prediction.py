import numpy as np
import pandas as pd


def create_edges_duplicate(edges):
    """Edges-ből csinál egy olyan Dataframe-et ahol az id1 oszlopban mindenki szerepel, az id2-ben pedig az összes szomszédjuk"""
 
    edges2=edges.copy(deep=True).reset_index().drop("index",axis=1)
    edges2.rename(columns={"smaller_id":"greater_id","greater_id":"smaller_id"},inplace=True) 
    
    edges_duplicate=edges.append(edges2).sort_values("smaller_id")
    edges_duplicate.rename(columns={"smaller_id":"id_1","greater_id":"id_2"},inplace=True)
    
    return edges_duplicate

def add_node_genders_to_edges(nodes, edges):
    """Az edges_duplicate df-hez hozzárendeli az id_2 (szomzédok) nemét és korát,
    illetve az id1 nemét, és hogy az adott id1 test vagy train """
    edges_w_genders = edges.merge(
        nodes[["user_id", "AGE", "gender"]].set_index("user_id"),
        how="left",
        left_on="id_2",
        right_index=True,
    )
    """Mivel a szomszédokhoz tartozik ez a nem, átnevezem gender_pairre."""
    edges_w_genders.rename(columns={"gender":"gender_pair"},inplace=True)
    edges_w_genders = edges_w_genders.merge(
        nodes[["user_id","gender","TRAIN_TEST"]].set_index("user_id"),
        how="left",
        left_on="id_1",
        right_index=True,
    )
    
    return edges_w_genders 

def only_test(edges_w_genders):
    """Csak azokat az elemeket tartja meg, ahol az id1 teszt + átnevezi nemeket"""
    edges_w_genders_test=edges_w_genders.loc[edges_w_genders["TRAIN_TEST"]=="TEST"]
    edges_w_genders_test["gender_pair"]=edges_w_genders_test["gender_pair"].replace({0.0: "woman", 1.0: "man"})
    
    return edges_w_genders_test


def predict_gender(nodes,edges):
    """"Elkészíti a becslést a test csúcsokra"""
    edges_duplicate=create_edges_duplicate(edges)
    edges_w_genders=add_node_genders_to_edges(nodes, edges_duplicate)
    edges_w_genders_test=only_test(edges_w_genders)
    
    """"Megszámolja, hogy személyekként hány nő és hány férfi párja van és elmenti 2 vektorba"""
    count_women=edges_w_genders_test.groupby('id_1')['gender_pair'].apply(lambda x: (x == 'woman').sum()).reset_index(name="woman")
    count_men=edges_w_genders_test.groupby('id_1')['gender_pair'].apply(lambda x: (x == 'man').sum()).reset_index(name="man")
    
    """Összemergeli a két táblát, így ebben már csak 3 oszlop lesz: id1, nő ismerősök száma, fiú ismerősök száma"""
    results = count_women.merge(
        count_men.set_index("id_1"),
        how="left",
        left_on="id_1",
        right_index=True,
    )
    
    """Beírja amelyikből több van (ha egyenlő, akkor legyen lány"""
    results["results"] = np.where(results["woman"]>=results["man"], 'woman', 'man')
    results.rename(columns={"id_1":"ID"},inplace=True)
    
    return results

    
    
    
    
    
    
