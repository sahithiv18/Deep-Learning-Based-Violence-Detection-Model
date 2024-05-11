# Deep-Learning-Based-Violence-Detection-Model

<h2>Objective:</h2>
<p>To build a deep learning based model that can detect Violence in a video</p>

<h2>About the project</h2>
<p> The model is a combination of deep learning techniques like CNN(Convolutional Neural Network) and LSTM(Long Short Term Memory). CNN for spatial extraction and LSTM for temporal feature extraction. Giving a video as an input. The model determines whether the video is "Violent" or "Non-Violent".</p>

<h2>Dataset</h2>
<p>The dataset I have used is Real Life Violence situations dataset:<a href= "https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset">here.</a> This dataset consists of 2000 videos, and is of 4gb in size.</p>

# Working
<p> The CNN model is a pre-trained model MobileNetV2 which was trained on a large set of videos for feature extraxtion.</p>
<h2>Examples:</h2>
<h2>Frame by frame prediction</h2></p>

    >input_video_file_path = r"C:\Users\sahit\Downloads\dataset\real life violence situations\Real Life Violence Dataset\Violence\V_981.mp4"
     predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
     show_pred_frames(output_video_file_path)

<img src="Violence Detection 1.jpeg">     
<p> Playing the actual video, the video of a man pushing a woman</p>

    >Play_Video(input_video_file_path)

<img src="Violence Detection 2.jpeg">

    >input_video_file_path = r"C:\Users\sahit\Downloads\dataset\real life violence situations\Real Life Violence Dataset\NonViolence\NV_98.mp4"
    predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
    show_pred_frames(output_video_file_path)

<img src="Violence Detection 3.jpeg">    
<img src="Violence Detection 4.jpeg">
<img src="Violence Detection 5.jpeg">

<h2>Prediction of Video</h2>

    >input_video_file_path = r"C:\Users\sahit\Downloads\dataset\real life violence situations\Real Life Violence Dataset\NonViolence\NV_997.mp4"
    predict_video(input_video_file_path, SEQUENCE_LENGTH)
    Play_Video(input_video_file_path)

<img src="Violence Detection 6.jpeg">

     >input_video_file_path = r"C:\Users\sahit\Downloads\dataset\real life violence situations\Real Life Violence Dataset\Violence\V_984.mp4"
     predict_video(input_video_file_path, SEQUENCE_LENGTH)
     Play_Video(input_video_file_path)

<img src="Violence Detection 7.jpeg">
<img src="Violence Detection 8.jpeg">
     
