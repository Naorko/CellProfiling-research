"""
This package is the interface for the dataset, responsible for loading and managing the dataset
    prepare_data.py - The package facade for loading a dataset
    tabular_dataset_with_ram.py - torch.utils.data.Dataset object that loads and filter plates on ram
    tabular_dataset_with_processed.py - torch.utils.data.Dataset object that loads a filtered plate
                                        from disk extracted by extract_processed_csvs.py
    create_tabular_metadata.py (standalone) - create a random partitioning of a tabular dataset to
                                              train\val\test and saves the proper details to csv
    extract_index.py (standalone) - extracts the index fields from an entire dataset
                                    and saves to a csv
"""

