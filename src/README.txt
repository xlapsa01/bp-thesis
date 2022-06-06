In order to run the script dependencies must be satisfied.
Dependency installation using YOLOv5 github for linux without CUDA:
	git clone https://github.com/ultralytics/yolov5  # clone
	cd yolov5
	pip install -r requirements.txt  # install

Dependency installation using YOLOv5 github for linux with CUDA:
	git clone https://github.com/ultralytics/yolov5  # clone
	cd yolov5
	pip install -r requirements.txt  # install
	pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

The following usage of the main.py implementing the proposed approach.
main.py usage:
	--v 	argument: takes a path to video
	--s 	argument: sets the scale ratio of the video resolution on which will be script applied
	--conf 	argument: sets the minimal confidence value of classification
	--ksize argument: sets the erosion kernel size, responsible for noise removal
	--a 	argument: sets minimal are of detection

64aug.pt contains the pre-trained custom YOLOv5 weights achieved with provided dataset.