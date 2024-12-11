# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/backhand/ --video-directory Data/Tennis/VIDEO_RGB/backhand/ 1>&2 > logs/tennis_backhand.log &
# echo "backhand: $!" >> logs/pids.txt

# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/backhand_slice/ --video-directory Data/Tennis/VIDEO_RGB/backhand_slice/ 1>&2 > logs/tennis_backhand_slice.log &
# echo "backhand_slice: $!" >> logs/pids.txt

# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/backhand_volley/ --video-directory Data/Tennis/VIDEO_RGB/backhand_volley/ 1>&2 > logs/tennis_backhand_volley.log 
# echo "backhand_volley: $!" >> logs/pids.txt

# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/backhand2hands/ --video-directory Data/Tennis/VIDEO_RGB/backhand2hands/ 1>&2 > logs/tennis_backhand2hands.log 
# echo "backhand2hands: $!" >> logs/pids.txt

# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/flat_service/ --video-directory Data/Tennis/VIDEO_RGB/flat_service/ 1>&2 > logs/tennis_flat_service.log 
# echo "flat_service: $!" >> logs/pids.txt

# nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/forehand_flat/ --video-directory Data/Tennis/VIDEO_RGB/forehand_flat/ 1>&2 > logs/tennis_forehand_flat.log 
# echo "forehand_flat: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/forehand_slice/ --video-directory Data/Tennis/VIDEO_RGB/forehand_slice/ 1>&2 > logs/tennis_forehand_slice.log 
echo "forehand_slice: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/forehand_volley/ --video-directory Data/Tennis/VIDEO_RGB/forehand_volley/ 1>&2 > logs/tennis_forehand_volley.log 
echo "forehand_volley: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/forehand_openstands/ --video-directory Data/Tennis/VIDEO_RGB/forehand_openstands/ 1>&2 > logs/tennis_forehand_openstands.log 
echo "forehand_openstands: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/kick_service/ --video-directory Data/Tennis/VIDEO_RGB/kick_service/ 1>&2 > logs/tennis_kick_service.log 
echo "kick_service: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/slice_service/ --video-directory Data/Tennis/VIDEO_RGB/slice_service/ 1>&2 > logs/tennis_slice_service.log 
echo "slice_service: $!" >> logs/pids.txt

nohup python PoseEstimation/pose_inferencer.py --output-directory outputs/tennis/smash/ --video-directory Data/Tennis/VIDEO_RGB/smash/ 1>&2 > logs/tennis_smash.log 
echo "smash: $!" >> logs/pids.txt