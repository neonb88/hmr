import sys

def SSSPG(log_path):
  """
    SSSPG stands for: stupidly_specific_string_processing_glue()
  """
  with open(log_path, 'r') as fp:
    lines=fp.readlines()
  # no newline
  dir_path=lines[0][:-1] # cut out newline
  img_name=lines[1].split(' ')[1] # ignore "CREATE" 1st-word
  uploaded_img_path=dir_path+img_name
  return uploaded_img_path

if __name__=="__main__":
  tmp='/home/n/Documents/code/old/hmr/tmp.txt'
  with open(tmp, 'w') as fp:
    fp.write(SSSPG(sys.argv[1]))
 
