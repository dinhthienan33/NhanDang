import os 
import cv2
import argparse
import utils 
import numpy as np 
from model.find_model import find_model
import time
def test(args):
    start_time=time.time()
    #testsets = ['test_c', 'test_b']

    #ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    ckpt_dir=args.ckpt_dir
    model, _ = find_model(args.model, 'test')
    
    epoch = model.load(ckpt_dir, epoch=args.ckpt_epoch)
    print(f'Loading {args.model} at EPOCH {epoch}!!')

    img_dir = os.path.join(args.dataset_dir)    
    #img_file_list = utils.get_image_file_list(img_dir)
    img_file_list=os.listdir(img_dir)
    img_file_list.sort()
    img_file_list=img_file_list

    for i, filename in enumerate(img_file_list, 1):
        img = cv2.imread(os.path.join(img_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if args.resize == 'square':
            square_img = cv2.resize(img, (512,512))
            input = utils.numpy2tensor(square_img)
            output = model.test_one_image(input)
        
        elif args.resize == 'expand':
            rows, cols = img.shape[:2]
            expand_img = utils.expand_size(img, 256)
            input = utils.numpy2tensor(expand_img)
            output = model.test_one_image(input)

            for title, output_img in output.items():
                output[title] = utils.restore_size(output_img, rows, cols)
        
        elif args.resize == 'original':
            input = utils.numpy2tensor(img)
            output = model.test_one_image(input)
            
        # save images 
        save_dir = os.path.join(args.save_dir, args.model, str(epoch) + '_' + args.resize)
        #utils.save_outputs(
         #   save_dir = save_dir,
          #  filename = f'{filename[:-4]}.png',
           # outputs = output,
            #max_display = 3
            #)

        save_output_dir = os.path.join(save_dir, 'output')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'{filename[:-4]}.png'), cv2.cvtColor(output['output'].squeeze() * 255, cv2.COLOR_RGB2BGR))
        
        print(f'{i}/{len(img_file_list)}:{filename}')
    
    print('Test Finished!!')
    print(time.time()-start_time)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog = 'DeRainDrop')             
    
    parser.add_argument('--model', default='proposed', type=str, dest='model')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, dest='ckpt_dir') 
    parser.add_argument('--ckpt_epoch', default=600, type=int, dest='ckpt_epoch')    
    parser.add_argument('--dataset_dir', default='dataset/', type=str, dest='dataset_dir')    
    parser.add_argument('--save_dir', default='../result/', type=str, dest='save_dir')
    parser.add_argument('--resize', default='original', type=str, dest='resize')
    
    args = parser.parse_args()
    test(args)




