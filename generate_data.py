import os

from dataset.arkit import DoDProcessor
from dataset.scannet import ScanNetProcessor

from dataset.scannet_scenes import all_scenes, training_scenes, evaluation_scenes, testing_scenes, sanity_scenes
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--root', type=str, help='path of the snapshot', default="DATA_ROOT/3dod")
    parser.add_argument('--output', type=str, help='path of the snapshot', default="DATA_ROOT/ARKit")
    parser.add_argument('--type', type=int, default=0, help="0: artkit; 1: scannet")
    parser.add_argument('--data_pkl', type=str, default=None)
    parser.add_argument('--scene', type=int, default=0, help="")
    parser.add_argument('--batch', type=int, default=1, help="")
    parser.add_argument('--stage', type=int, default=2,
                        help='1 process the single frames; 2 combine information of all frames for training and select key frames for testing environment; 3 output the downsampled frames')
    args = parser.parse_args()

    if args.type == 0:
        if args.stage == 2:
            args.scene = 0
            args.batch = 1
        # output_dir = "/home/jing/Documents/work/data/ThreeDoDSim"
        arkit = DoDProcessor(display=False, root=args.root, scene_ind=args.scene, batch=args.batch,
                             output_root=args.output, data_pkl=args.data_pkl)

        if args.stage == 1:
            arkit.run_scenes()
        elif args.stage == 2:
            training_scenarios, evaluation_scenarios, test_scenarios = arkit.get_scenes()

            for scenes, name in [[training_scenarios, "training"],
                                 [evaluation_scenarios, "evaluation"],
                                 [test_scenarios, "testing"]]:
                arkit.combine_info(scenes=scenes, name=name)
            arkit.process_key_frames(name="testing")
        else:
            arkit.process_down_sample(batch=args.scene, output_dir=args.output, total_batch=args.batch)
    else:
        scannet = ScanNetProcessor(display=False, root=args.root, output_root=args.output)
        all_scenes.sort()
        if args.stage == 1:
            if args.batch == 1:
                scannet.run_scenes(scenes=all_scenes)
                # scannet.run_scenes(scenes=sanity_scenes)
            else:
                each_batch = int(len(all_scenes) / args.batch)
                if args.scene == 0:
                    scannet.run_scenes(scenes=all_scenes[:each_batch])
                elif args.scene == args.batch - 1:
                    scannet.run_scenes(scenes=all_scenes[(args.batch - 1) * each_batch:])
                else:
                    scannet.run_scenes(scenes=all_scenes[each_batch * args.scene:each_batch * (1 + args.scene)])
        elif args.stage == 2:
            for scenes, name in [[training_scenes, "training"],
                                 [evaluation_scenes, "evaluation"],
                                 [testing_scenes, "testing"]]:
                scannet.combine_info(scenes=scenes, name=name)
            scannet.process_key_frames(name="testing")
        else:
            scannet.process_down_sample(batch=args.scene, output_dir=args.output, total_batch=args.batch)