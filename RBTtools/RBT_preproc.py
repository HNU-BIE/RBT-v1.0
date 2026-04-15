import ants


def extract_CTskull():
    img=ants.image_read('CT.nii.gz')
    seg=ants.otsu_segmentation(img,7)
    seg.to_file('/CT_skull_segmentation.nii.gz')
    mask=img-ants.mask_image(img,seg,[1,2,3])
    mask_data=mask.numpy()
    mask_data[mask_data>0]=1
    mask[:,:,:]=mask_data
    mask_=ants.morphology(mask,'dilate',2)
    mask_=ants.morphology(mask_,'close',2)
    img_=ants.mask_image(img,mask_)
    img_.to_file('CT_skull.nii.gz')

def CT_aligntoT2w():
    ct = ants.image_read('CT_skull.nii.gz')
    img = ants.image_read('T2w_exvivo.nii.gz')
    mask=ants.get_mask(img)
    img=img-ants.mask_image(img,mask)
    ct_=ants.resample_image_to_target(ct,img)
    t = ants.registration(img, ct_, type_of_transform='Similarity',aff_metric='mattes')
    ct_ = ants.apply_transforms(img, ct_, t['fwdtransforms'],'bSpline')
    ct_.to_file('CT_aligntoT2w.nii.gz')

def aligntoTemplate():
    ct = ants.image_read('CT_aligntoT2w.nii.gz')
    img = ants.image_read('T2w_exvivo.nii.gz')
    tmp=ants.image_read('RBT_Template_T2w.nii.gz')
    t=ants.registration(tmp,img,'SyN',syn_metric='mattes',aff_metric='GC',reg_iterations=(400,200,100),flow_sigma=3)
    img_ = ants.apply_transforms(tmp, img, t['fwdtransforms'], 'bSpline')
    ct_ = ants.apply_transforms(tmp, ct, t['fwdtransforms'], 'bSpline')
    img_.to_file('T2w_inTMP.nii.gz')
    ct_.to_file('CT_inTMP.nii.gz')