import os
import pandas as pd

def rename_KDD_labels(path, filename, name_postfix, rename_fn):
    dataset = pd.read_csv(os.path.join(path,filename), sep=",")
    dataset["attack"] = dataset["attack"].apply(rename_fn)
    
    new_filename = 'KDDTest' if 'test' in filename.lower() else 'KDDTrain'
    new_filename = new_filename + '_' + name_postfix + '.csv'
    full_path = os.path.join(path,new_filename)

    print('Saving to ', full_path)
    dataset.to_csv(full_path, index=False)

def convert_to_2_class(path, filename):
    def rename_fn(attack):
        return 'normal' if attack == 'normal' else 'attack'

    rename_KDD_labels(path, filename, '2c', rename_fn)

def convert_to_5_class(path, filename):
    attack_types_dict = {
        'u2r':['ps','xterm','sqlattack','buffer_overflow','loadmodule','perl','rootkit'],
        'r2l':['named','sendmail','httptunnel','snmpgetattack','snmpguess', 'xsnoop', 'xlock','ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster'],
        'probe':['saint','mscan','ipsweep','nmap','portsweep','satan'],
        'dos':['mailbomb','worm','processtable','udpstorm','apache2','back','land','neptune','pod','smurf','teardrop']
    }

    def rename_fn(attack):
        for key in attack_types_dict:
            if attack in attack_types_dict[key]:
                return key
        return 'normal'

    rename_KDD_labels(path, filename, '5c', rename_fn)


def reduce_to_general_type(path, filename):
    attack_types_dict = {
        'u2r':['ps','xterm','sqlattack','buffer_overflow','loadmodule','perl','rootkit'],
        'r2l':['named','sendmail','httptunnel','snmpgetattack','snmpguess', 'xsnoop', 'xlock','ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster'],
        'probe':['saint','mscan','ipsweep','nmap','portsweep','satan'],
        'dos':['mailbomb','worm','processtable','udpstorm','apache2','back','land','neptune','pod','smurf','teardrop']
    }

    dataset = pd.read_csv(os.path.join(path,filename), sep=",")

    for key in attack_types_dict: 

        attacks_and_normal = attack_types_dict[key] + ['normal']
        new_ds = dataset[dataset.attack.isin(attacks_and_normal)]

        new_filename = 'KDDTest' if 'test' in filename.lower() else 'KDDTrain'
        new_filename = new_filename + '_' + key + '.csv'
        full_path = os.path.join(path,new_filename)
        print('Saving to ', full_path)
        new_ds.to_csv(full_path, index=False)