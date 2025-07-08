import pandas as pd
pd.set_option('display.max_rows', 9999)
import numpy as np
from modello_base import ModelloBase
from scipy.stats import chi2_contingency, contingency
import matplotlib.pyplot as plt


class ModelloScacchi(ModelloBase):
    def __init__(self,dataset_path):
        self.dataframe=pd.read_csv(dataset_path)
        self.df_sistemato=self.sistemazione_dataframe()


    def sistemazione_dataframe(self):
        # 1. copia del dataframe
        df_sistemato = self.dataframe.copy()
        #2.drop colonne inutili all'analisi
        colonne_da_droppare=["id","created_at","last_move_at","white_id","black_id","opening_eco"]
        df_sistemato=df_sistemato.drop(colonne_da_droppare,axis=1)
        #3. aggiunta colonna category time in base a increment_code
        df_sistemato["minuti"]=df_sistemato["increment_code"].str.split("+").str[0].astype(int)
        df_sistemato["category_time"]=pd.cut(df_sistemato["minuti"],bins=[-1,5,20,300],labels=["short","mean","long"])
        df_sistemato=df_sistemato.drop(["minuti",],axis=1)
        #4 generalizzazione categorie aperture
        condizioni=[
            df_sistemato["opening_name"].str.contains("Slav Defense",case=False),
            df_sistemato["opening_name"].str.contains("Nimzowitsch Defense", case=False),
            df_sistemato["opening_name"].str.contains("King's Pawn Game", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Pawn Game", case=False),
            df_sistemato["opening_name"].str.contains("Philidor Defense", case=False),
            df_sistemato["opening_name"].str.contains("Blackmar-Diemer Gambit", case=False),
            df_sistemato["opening_name"].str.contains("Italian Game", case=False),
            df_sistemato["opening_name"].str.contains("Scandinavian Defense", case=False),
            df_sistemato["opening_name"].str.contains("Van't Kruijs Opening", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Knight Variation", case=False),
            df_sistemato["opening_name"].str.contains("Four Knights Game", case=False),
            df_sistemato["opening_name"].str.contains("Horwitz Defense", case=False),
            df_sistemato["opening_name"].str.contains("English Opening: King's English Variation", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Smith-Morra Gambit", case=False),
            df_sistemato["opening_name"].str.contains("Scotch Game", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Refused: Marshall Defense", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Accepted: Central Variation", case=False),
            df_sistemato["opening_name"].str.contains("Indian Game", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Dragon Variation", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Closed Variation", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Normal Variation", case=False),
            df_sistemato["opening_name"].str.contains("Dutch Defense", case=False),
            df_sistemato["opening_name"].str.contains("Zukertort Opening", case=False),
            df_sistemato["opening_name"].str.contains("Vienna Game", case=False),
            df_sistemato["opening_name"].str.contains("Modern Defense", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Pawn Game: Mason Attack", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Advance", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Exchange Variation", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Pawn Game: London System", case=False),
            df_sistemato["opening_name"].str.contains("Crab Opening", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Winawer Variation", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Indian Defense", case=False),
            df_sistemato["opening_name"].str.contains("Gruenfeld Defense", case=False),
            df_sistemato["opening_name"].str.contains("French Defense: Rubinstein Variation", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Declined: Semi-Tarrasch", case=False),
            df_sistemato["opening_name"].str.contains("Yusupov-Rubinstein System", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Declined: Traditional Variation", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Accepted: Old Variation", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Declined: Queen's Knight Variation", case=False),
            df_sistemato["opening_name"].str.contains("Ruy Lopez: Classical Variation", case=False),
            df_sistemato["opening_name"].str.contains("Bishop's Opening: Ponziani Gambit", case=False),
            df_sistemato["opening_name"].str.contains("Benoni Defense", case=False),
            df_sistemato["opening_name"].str.contains("King's Indian Attack", case=False),
            df_sistemato["opening_name"].str.contains("Alekhine Defense", case=False),
            df_sistemato["opening_name"].str.contains("Caro-Kann Defense", case=False),
            df_sistemato["opening_name"].str.contains("Goldsmith Defense", case=False),
            df_sistemato["opening_name"].str.contains("Nimzo-Indian Defense", case=False),
            df_sistemato["opening_name"].str.contains("Bogo-Indian Defense", case=False),
            df_sistemato["opening_name"].str.contains("King's Knight Opening", case=False),
            df_sistemato["opening_name"].str.contains("Vienna Game", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Hyperaccelerated Dragon", case=False),
            df_sistemato["opening_name"].str.contains("Ruy Lopez: Berlin Defense", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Najdorf", case=False),
            df_sistemato["opening_name"].str.contains("Amar Opening", case=False),
            df_sistemato["opening_name"].str.contains("Ruy Lopez: Morphy Defense", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense: Alapin Variation", case=False),
            df_sistemato["opening_name"].str.contains("Center Game", case=False),
            df_sistemato["opening_name"].str.contains("Englund Gambit", case=False),
            df_sistemato["opening_name"].str.contains("Petrov's Defense", case=False),
            df_sistemato["opening_name"].str.contains("Russian Game", case=False),
            df_sistemato["opening_name"].str.contains("Pirc Defense", case=False),
            df_sistemato["opening_name"].str.contains("Owen Defense", case=False),
            df_sistemato["opening_name"].str.contains("Semi-Slav Defense", case=False),
            df_sistemato["opening_name"].str.contains("Bird Opening", case=False),
            df_sistemato["opening_name"].str.contains("Ponziani Opening", case=False),
            df_sistemato["opening_name"].str.contains("East Indian Defense", case=False),
            df_sistemato["opening_name"].str.contains("Reti Opening", case=False),
            df_sistemato["opening_name"].str.contains("Nimzo-Larsen Attack", case=False),
            df_sistemato["opening_name"].str.contains("Torre Attack", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Pawn", case=False),
            df_sistemato["opening_name"].str.contains("Sicilian Defense", case=False),
            df_sistemato["opening_name"].str.contains("French Defense", case=False),
            df_sistemato["opening_name"].str.contains("English Opening", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Declined", case=False),
            df_sistemato["opening_name"].str.contains("Queen's Gambit Accepted", case=False),
            df_sistemato["opening_name"].str.contains("Ruy Lopez", case=False)
        ]
        categorie= [
            "Slav Defense",
            "Nimzowitsch Defense",
            "King's Pawn Game",
            "Queen's Pawn Game",
            "Philidor Defense",
            "Blackmar-Diemer Gambit",
            "Italian Game",
            "Scandinavian Defense",
            "Van't Kruijs Opening",
            "French Defense: Knight Variation",
            "Four Knights Game",
            "Horwitz Defense",
            "English Opening: King's English Variation",
            "Sicilian Defense: Smith-Morra Gambit",
            "Scotch Game",
            "Queen's Gambit Refused: Marshall Defense",
            "Queen's Gambit Accepted: Central Variation",
            "Indian Game",
            "Sicilian Defense: Dragon Variation",
            "Sicilian Defense: Closed Variation",
            "French Defense: Normal Variation",
            "Dutch Defense",
            "Zukertort Opening",
            "Vienna Game",
            "Modern Defense",
            "Queen's Pawn Game: Mason Attack",
            "French Defense: Advance",
            "French Defense: Exchange Variation",
            "Queen's Pawn Game: London System",
            "Crab Opening",
            "French Defense: Winawer Variation",
            "Queen's Indian Defense",
            "Gruenfeld Defense",
            "French Defense: Rubinstein Variation",
            "Queen's Gambit Declined: Semi-Tarrasch",
            "Yusupov-Rubinstein System",
            "Queen's Gambit Declined: Traditional Variation",
            "Queen's Gambit Accepted: Old Variation",
            "Queen's Gambit Declined: Queen's Knight Variation",
            "Ruy Lopez: Classical Variation",
            "Bishop's Opening: Ponziani Gambit",
            "Benoni Defense",
            "King's Indian Attack",
            "Alekhine Defense",
            "Caro-Kann Defense",
            "Goldsmith Defense",
            "Nimzo-Indian Defense",
            "Bogo-Indian Defense",
            "King's Knight Opening",
            "Vienna Game",
            "Sicilian Defense: Hyperaccelerated Dragon",
            "Ruy Lopez: Berlin Defense",
            "Sicilian Defense: Najdorf",
            "Amar Opening",
            "Ruy Lopez: Morphy Defense",
            "Sicilian Defense: Alapin Variation",
            "Center Game",
            "Englund Gambit",
            "Petrov's Defense",
            "Russian Game",
            "Pirc Defense",
            "Owen Defense",
            "Semi-Slav Defense",
            "Bird Opening",
            "Ponziani Opening",
            "East Indian Defense",
            "Reti Opening",
            "Nimzo-Larsen Attack",
            "Torre Attack",
            "Queen's Pawn",
            "Sicilian Defense",
            "French Defense",
            "English Opening",
            "Queen's Gambit Declined",
            "Queen's Gambit Accepted",
            "Ruy Lopez"
        ]
        df_sistemato["opening"]=np.select(condizioni,categorie,default="Altro")
        #5. creazione categorie di differenza rating per rendere più comprensibili i risultati dei match
        df_sistemato["rating_diff"]=df_sistemato["white_rating"] - df_sistemato["black_rating"]
        rating_bins=[-2000,-200,-50,50,200,2000]
        rating_labels=["Molto favorito Nero","Favorito Nero","Equilibrato","Favorito Bianco","Molto favorito Bianco"]
        df_sistemato["diff_group"]=pd.cut(df_sistemato["rating_diff"],bins=rating_bins,labels=rating_labels)
        # grouped=df_sistemato.groupby(["diff_group","winner"]).size().unstack(fill_value=0)
        # print(grouped)

        return df_sistemato

    def tabella_di_contingenza(self,target):
        tabella= self.df_sistemato[target].value_counts()
        return tabella

    def salvataggio_csv(self):
        self.df_sistemato.to_csv("../dataset_ripulito/games_clean.csv",index=False)




modello= ModelloScacchi("../dataset/games.csv")
modello.analisi_generali(modello.df_sistemato)
modello.analisi_valori_univoci(modello.df_sistemato,["turns","moves","white_rating","black_rating"])
#print(modello.tabella_di_contingenza("opening"))
modello.salvataggio_csv()