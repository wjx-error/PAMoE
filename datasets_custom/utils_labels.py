def trans_labels(lb, args):
    if args.task == 'survival':
        return int(lb)
    elif args.task == 'staging':
        return staging_label(lb)
    elif args.task == 'subtype':
        return subtype_label(lb, args.project_name)
    else:
        raise NotImplementedError


def staging_label(lb):
    if lb in ['Stage I', 'Stage IA', 'Stage IB']:
        label = 0
    elif lb in ['Stage IIA', 'Stage IIB', 'Stage II', 'Stage IIC']:
        label = 1
    elif lb in ['Stage IIIB', 'Stage IIIC', 'Stage III', 'Stage IIIA']:
        label = 2
    elif lb in ['Stage IV', 'Stage IVA', 'Stage IVB']:
        label = 3
    else:
        return None
        # raise ValueError("Undefined label")
    return label


subtype_dict = {
    'LGG': ['Astrocytoma, anaplastic', 'Oligodendroglioma, anaplastic', 'Mixed glioma'],
    'BRCA': ['Lobular carcinoma', 'Infiltrating duct carcinoma'],
    'COAD': ['Mucinous adenocarcinoma', 'Adenocarcinoma']
}


def subtype_label(lb, project_nm):
    # print(lb)
    list_label = list(subtype_dict.keys())
    assert project_nm in list_label
    label = 0
    for x in subtype_dict[project_nm]:
        if lb in x:
            return label
        label += 1
    return None
