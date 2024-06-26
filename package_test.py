from Unimol_2_NMR_fix.train import NMR_Fix_Master

nfm = NMR_Fix_Master(datahub_config = {
                        'data_path' : '/vepfs/fs_users/ycjin/Delta-ML-Framework/Unimol_2_NMR_fix/example/raw_data/ml_pbe0_pcSseg-2_c.csv',
                        'task' : 'all',
                        'save_dir' : '/vepfs/fs_users/ycjin/Delta-ML-Framework/Unimol_2_NMR_fix/example/process_data',
                        'desc' : ['unimol',],
                        'finetune': True,
                        'if_process':True, 
                        'save_processed_data2csv':True,
                        'clip_size':0.875,
                        'structure_level': 'atom', # use 'atom' or 'molecule' or 'system'
                        'structure_source': 'files', # if .xyz or .mol use 'files'; if smiles us 'smiles'
                        'structure_cols': 'filepath', # use structure_base col-name in your csv
                        'structure_atom': 'atom', # use atom_id col-name in your csv
                        'feature_cols': [], # all feature cols in your csv; or use None to edit drop_cols; for case only use desc use []
                        'label_cols': ['shift_high-low',], # all label cols in your csv
                        'drop_cols': ['compound','structure','shift_low'], # all useless cols in your csv
                        'train':True,
                     },

                     trainer_config = {
                        'max_epoch':1000,
                        'patience':20,
                        'batch_size':32,
                     },

                     modelhub_config = {
                        'train_size':0.875,
                        'model_name':'MLPModel',
                     })

nfm.run()