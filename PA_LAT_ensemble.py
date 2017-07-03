import numpy as np
from sklearn import metrics
from work_v2 import get_model, get_test_datagen, count_files
import itertools

groups = ['M_Adult', 'F_Adult']
positions = ['PA', 'LAT']
models = ['resnet50', 'vgg16', 'vgg19']
tops = ['waya', 'chollet', 'linear']


def scores_from_model_top(group, position, model, top):
    weights_path = 'models/{group}/{position}/{model}/{top}/top_trained.h5'.format(position=position, group=group,
                                                                                   model=model, top=top)
    test_path = 'data/{position}_256/{group}/test/'.format(position=position, group=group)
    full_model = get_model(model, top)
    full_model.load_weights(weights_path)
    test_datagen = get_test_datagen(model)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        # target_size=(224, 224),
        reader_config={'target_mode': 'RGB', 'target_size': target_size},
        batch_size=batch_size,
        shuffle=False)
    preds = full_model.predict_generator(
        generator=test_generator,
        steps=n_steps_test,
        workers=4,
        verbose=1)
    scores = preds[:, 1]
    return scores


def ensemble_roc_auc(y, scores, combination):
    ensemble_score = np.mean([scores[combination[0]], scores[combination[1]]], axis=0)
    fpr, tpr, thresholds = metrics.roc_curve(y, ensemble_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def ensemble_precision_recall(y, scores, combination):
    ensemble_score = np.mean([scores[combination[0]], scores[combination[1]]], axis=0)
    precision, recall, thresholds = metrics.precision_recall_curve(y, ensemble_score)
    return precision, recall, thresholds


scores = [scores_from_model_top(group, position, model, top) for group, position, model, top in
          itertools.product(groups, positions, models, tops)]
np.save('scores.npy', scores)
# test_path = 'data/{position}_224/{group}/test/'.format(position=position, group=group)
# num_files = sum(os.path.isfile(os.path.join(test_path, f)) for f in os.listdir(test_path))
# y = num_files/2 * [0] + num_files/2 * [1]
# combinations = [comb for comb in itertools.combinations(range(9), 2)]
# aucs = [ensemble_roc_auc(y, scores, comb) for comb in combinations]
# comb_names = [(m1, m2) for (m1, m2) in itertools.combinations(itertools.product(models, tops), 2)]
