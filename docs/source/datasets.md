# Autonomous Driving Dataset

[The Waymo Open Dataset (WOD)](https://waymo.com/open/download/) contains 
*   Perception Dataset, latest version: [v2.0.1 files](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1) and [v1.4.3 files](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3). V2.0 dataset is in modular format (Apache Parquet column-oriented file format), enabling users to selectively download only the components they need. This format separates the data into multiple tables, allowing users to selectively download the portion of the dataset needed for their specific use case. V1.0 is TFrecord files (Frame binary protocol buffers serialized into tfrecords files). Column values in v2-supported components is the same as corresponding proto fields in the v1.4.2.
*   Motion Dataset: [v1.2.1 files](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_1)
*   Github for technical specs: [link](https://github.com/waymo-research/waymo-open-dataset)
*   [V2 Dataset Tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb)