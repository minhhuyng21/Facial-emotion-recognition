from deepface import DeepFace

objs = DeepFace.analyze(
  img_path = "DSC_6883.jpg", 
  actions = ['emotion'],
)

print(objs)