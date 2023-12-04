Bước 0:
	tại thư mục project:
	mkdir data
	cd data
	mkdir INDEX; mkdir npy

	file all_images.out	:
		Lashinbang_data/Images_300_399/317/203510243317_S2.jpg
		Lashinbang_data/Images_300_399/317/202010095317_S2.jpg
		Lashinbang_data/Images_300_399/317/203510593317_L2.jpg
		Lashinbang_data/Images_300_399/317/201310021317_S.jpg
		Lashinbang_data/Images_300_399/317/204010193317_L.jpg
		Lashinbang_data/Images_300_399/317/205010322317_L2.jpg
		Lashinbang_data/Images_300_399/317/205010289317_L2.jpg
		Lashinbang_data/Images_300_399/317/203510198317_S2.jpg
		Lashinbang_data/Images_300_399/317/207310029317_L2.jpg
		...
Bước 1:
	python generate_features.py

	output là các file npy chứa các feature vectors trong ./data/npy, mỗi file max size 10000 vectors

Bước 2: 
	Merge một file feature khoảng 2M vectors để training index
	python generate_training_data.py

	output là file trained.index trong ./data

Bước 3:
	từ feature vectors, tạo faiss index và paths tương ứng
	python npy2faiss_index.py
