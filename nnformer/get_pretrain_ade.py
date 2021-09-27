import torch


p1=['patch_embed.proj',"relative_position","downsample",'decode_head','auxiliary_head']
p2=['layers.2.blocks.%d'%i for i in range(18)[2:]]
#if you use the imagenet pretrain weight, the p1 and p2 are as follows
'''
p1=['patch_embed.proj',"relative_position","downsample",'attn_mask','head']
p2=['layers.2.blocks.%d'%i for i in range(18)[2:]]
'''
pop_list=p1+p2


def remove():
    #input the pretrain weight of the swin transformer
    pretrained_dict = torch.load('./upernet_swin_small_patch4_window7_512x512.pth',map_location='cpu')
    weight=pretrained_dict['state_dict']
    
    for i in list(weight.keys()):
        for j in pop_list:
            if j in i:
                weight.pop(i)
                break
    for i in weight:
        print(i)
    return weight

def rename(weight):
    up_weight=weight.copy()
    for i in list(weight.keys()):
        weight.update({i.replace('backbone','model_down'):weight.pop(i)})
        
  
    for i in list(up_weight.keys()):
        up_weight.update({i.replace('backbone','encoder'):up_weight.pop(i)})
        
    weight.update(up_weight)
    #del unnecessary key in up_weight
    #the encoer name of our model is model_down
    #the decoder name of our model is encoer 
    #haaa,I'm sorry for the mistake
    weight.pop('encoder.patch_embed.norm.weight')
    weight.pop('encoder.patch_embed.norm.bias')
    for i in list(weight):
        if 'encoder.layers.3' in i or 'encoder.norm' in i:
            weight.pop(i)

    return weight

     
if __name__=='__main__':
    weight=remove()
    weight=rename(weight)
    #input one checkpoint of your model
    our_dict = torch.load("/model_best.model",map_location='cpu')
    our_weight=our_dict['state_dict']

    for i in weight:
        our_weight[i]=weight[i]
    our_dict['state_dict']=our_weight
    torch.save(our_dict,'xxx.model')
    
    
    
    