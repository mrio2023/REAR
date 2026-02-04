from components import main_Embedding
def run():
    main_Embedding.main(dataset_name="AscendEXHacker", graph_type="normal")
    main_Embedding.main(dataset_name="PlusTokenPonzi", graph_type="normal")
    main_Embedding.main(dataset_name="elliptic_txs", graph_type="elliptic")
   
    

run()