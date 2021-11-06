 import glob
import ffmpeg
import random
import argparse
import librosa
import pydiffvg
import clip
import wav2clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Create the parser
    wav2draw_parser = argparse.ArgumentParser(description="Doodles generation using Wav2CLIP+CLIPDraw")

    # Add the arguments
    wav2draw_parser.add_argument("-ap", "--audio_prompts", type=str, help="Audio file as prompts", default="sample.wav", dest="audio_prompts",
    )
    wav2draw_parser.add_argument(
        "-ai", "--audio_index", type=int, default=None, dest="audio_index", help="audio index"
    )
    wav2draw_parser.add_argument(
        "-aframe", "--audio_frame_length", type=int, default=None, dest="audio_frame_length", help="audio hop length"
    )
    wav2draw_parser.add_argument(
        "-ahop", "--audio_hop_length", type=int, default=None, dest="audio_hop_length", help="audio hop length"
    )
    wav2draw_parser.add_argument(
        "-asf", "--audio_sampling_freq", type=int, default=16000, dest="audio_sampling_freq"
    )
    wav2draw_parser.add_argument(
        "-np", "--num_paths", type=int, default=256, dest="num_paths", help="total number of strokes"
    )
    wav2draw_parser.add_argument(
        "-mw", "--max_width", type=int, default=50, dest="max_width", help="max length of a stroke"
    )
    wav2draw_parser.add_argument(
        "-ni", "--num_iter", type=int, default=1000, dest="num_iter", help="total number of iterations for optimizing the drawings"
    )

    wav2draw_parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="Device to use (CPU or GPU)",
        default="cuda:0",
        dest="device",
    )
        
    # Execute the parse_args() method
    args = wav2draw_parser.parse_args()
    
    # Load Wav2CLIP model
    wav2clip_model = wav2clip.get_model(device=args.device)
    clip_model, _ = clip.load('ViT-B/32', args.device, jit=False)

    if not args.device == "cpu":
        assert torch.cuda.is_available(), "No GPU found"

    device = args.device

    audio, sr = librosa.load(args.audio_prompts, sr=args.audio_sampling_freq)
    
    if args.audio_index and args.audio_frame_length and args.audio_hop_length:
        start = args.audio_hop_length * args.audio_index
        audio = audio[start : start + args.audio_frame_length]

    embed = torch.from_numpy(wav2clip.embed_audio(audio, wav2clip_model)).to(device)

    # Check device for pydiffvg
    pydiffvg.set_device(torch.device(args.device))

    # Define canvas size
    canvas_width, canvas_height = 224, 224
    gamma = 1.0

    # Image Augmentation Transformation
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    ])

    use_normalized_clip = False

    if use_normalized_clip:
        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Initialize Random Curves
    shapes = []
    shape_groups = []
    for i in range(args.num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)

    # Just some diffvg setup
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Run the inner optimization loop
    for t in range(args.num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(args.num_iter * 0.5):
            for g in points_optim.param_groups:
                g['lr'] = 0.4
        if t == int(args.num_iter * 0.75):
            for g in points_optim.param_groups:
                g['lr'] = 0.1
            
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        if t % 5 == 0:
            pydiffvg.imwrite(img.cpu(), './res/iter_{}.png'.format(int(t/5)), gamma=gamma)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

        loss = 0
        NUM_AUGS = 4
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        image_features = clip_model.encode_image(im_batch)
        for n in range(NUM_AUGS):
            loss -= torch.cosine_similarity(embed, image_features[n:n+1], dim=1)
                
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        width_optim.step()
        color_optim.step()
            
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, args.max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)


    input_still = ffmpeg.input(glob.glob("res/iter_*.png")[-1])
    input_audio = ffmpeg.input(args.audio_prompts)

    (
        ffmpeg
        .concat(input_still, input_audio, v=1, a=1)
        .output("output.mp4")
        .run(overwrite_output=True)
    )

    
