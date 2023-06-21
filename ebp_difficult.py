"""
I have a file indicating if an image is difficult for a class from VOC2007. the files structure is image_id <0 if easy,1 if difficult> for the 20 classes. here are 2 example rows from the file.
000001	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0
000002	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0

I would like to create a class called VOCDifficult which reads in thi file. it has a function called is_difficult which takes in an image_id and target_id and returns if that image is difficult for that class.
"""
import lzma
import pickle
import os
import glob
VOC_DIFFICULT_TXT = '/root/evaluate-saliency-4/GPNN/benchmark/dataset/pointing_game_ebp_voc07_difficult.txt'
class VOCDifficult:
    def __init__(self, file_path=VOC_DIFFICULT_TXT):
        self.difficult_data = {}
        self.load_difficult_data(file_path)

    def load_difficult_data(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                image_id = line[0]
                difficult_labels = [int(label) for label in line[1:]]
                self.difficult_data[image_id] = difficult_labels

    def is_difficult(self, image_id, target_id):
        if image_id not in self.difficult_data:
            return False  # Image ID not found in the data

        difficult_labels = self.difficult_data[image_id]

        if target_id < 0 or target_id >= len(difficult_labels):
            return False  # Invalid target ID

        return difficult_labels[target_id] == 1

def test():
    # Usage example:
    # DIFFICULT_TXT = '/root/evaluate-saliency-4/GPNN/benchmark/dataset/pointing_game_ebp_voc07_difficult.txt'
    voc_difficult = VOCDifficult(VOC_DIFFICULT_TXT)
    image_id = '000001'
    target_id = 11  # Class index (0-19)

    is_difficult = voc_difficult.is_difficult(image_id, target_id)
    print(f"Is image {image_id} difficult for class {target_id}? {is_difficult}")
    target_id = 14
    is_difficult = voc_difficult.is_difficult(image_id, target_id)
    print(f"Is image {image_id} difficult for class {target_id}? {is_difficult}")
"""
i have data saved for the saliency map computed for pascal images for applicable classes. example of some entries of the data
    {'000001':[{classname:'dog','target_id':11,'is_correctly_predicted':False},{classname:'person','target_id':14,'is_correctly_predicted':True},]
    '000002':[{classname:'car','target_id':6,'is_correctly_predicted':True}]
    }
i want to iterate through entries of the data, with 2 loops. the outer is across image_ids, the inner is across the different objects for that image. i will pass in the target_id and image_id to the is_difficult function of an object of the VOCDifficult class, and get back if it is difficult of not

"""
def test_with_saved_data():
    difficult_data = {
        '000001': [
            {'classname': 'dog', 'target_id': 11, 'is_correctly_predicted': False},
            {'classname': 'person', 'target_id': 14, 'is_correctly_predicted': True},
        ],
        '000002': [
            {'classname': 'car', 'target_id': 6, 'is_correctly_predicted': True}
        ]
    }

    voc_difficult = VOCDifficult(VOC_DIFFICULT_TXT)  # Assuming you have defined VOCDifficult class

    for image_id, objects in difficult_data.items():
        print(f"Image ID: {image_id}")
        for obj in objects:
            target_id = obj['target_id']
            is_difficult = voc_difficult.is_difficult(image_id, target_id)
            classname = obj['classname']
            is_correctly_predicted = obj['is_correctly_predicted']
            
            print(f" - Class: {classname}")
            print(f"   Target ID: {target_id}")
            print(f"   Is Difficult: {is_difficult}")
            print(f"   Is Correctly Predicted: {is_correctly_predicted}")  
from benchmark import settings
def aggregate_pointing(
    dataset,
    methodname,
    modelname,
    metricname = 'pointing_game'):
    methoddir = os.path.join(settings.METRICS_DIR_librecam,f'{dataset}-{metricname}-{methodname}-{modelname}')
    assert os.path.isdir(methoddir)
    imdirs = glob.glob(os.path.join(methoddir,'*/'))
    imdirs = list(sorted(imdirs))
    all_pointing = []
    difficult_pointing = []
    difficult_obj = VOCDifficult(VOC_DIFFICULT_TXT)
    
    for imdir in imdirs:
        print(os.listdir(imdir))
        pklnames = glob.glob(os.path.join(imdir,'*.xz'))
        imroot = os.path.basename(
        imdir.rstrip(os.path.sep))
        import ipdb;ipdb.set_trace()
        for pklname in pklnames:
            with lzma.open(pklname,'rb') as f:
                loaded = pickle.load(f) 
            """
            loaded = {
                'target_id':14,
                'hit':1,
            }
            """
            all_pointing.append(
                loaded['hit']
            )
            if difficult_obj.is_difficult(imroot,loaded['target_id']):
                difficult_pointing.append(loaded['hit'])
                import ipdb;ipdb.set_trace()
        # with open(os.path.join(imroot_dir)):
        #     pass
        # break
    import ipdb;ipdb.set_trace()
    pass
if __name__ == '__main__':
    # test()
    # test_with_saved_data()
    aggregate_pointing('pascal','gradcam','resnet50')
    pass