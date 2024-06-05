if __name__ == '__main__':
  from multiprocessing import freeze_support
  freeze_support()
  
  import fastbook
  fastbook.setup_book()
  from fastbook import *
  from fastai.vision.all import *
  from fastai.vision.widgets import *
  from matplotlib import *


  lung_types = ('NORMAL', 'VIRUS', 'BACTERIA')
  path = Path('D:\\pneumonia\\Dataset\\out')

  lungs = DataBlock(
      blocks=(ImageBlock, CategoryBlock),
      get_items=get_image_files,
      splitter=RandomSplitter(valid_pct=0.2, seed=42),
      get_y=parent_label,
      item_tfms=Resize(128)
  )

  dls = lungs.dataloaders(path, num_workers=0)  # Set num_workers=0 to disable parallel loading

  learn = vision_learner(dls, resnet50, metrics=accuracy)
  dls.show_batch(rows=5, cols=2)  
  learn.fine_tune(1)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.save('model.pkl')
