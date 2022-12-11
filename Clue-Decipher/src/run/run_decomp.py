import argparse
from modeling.seq2seq_decomp import *
from utils.utils import *


def main(args):
    # logger
    seed_everything(args.seed)
    log_name = '../../log/' + args.file_name
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s]- %(message)s')

    # set print format
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # set log
    fh = logging.FileHandler(log_name, encoding='utf8', mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logging.info(args)

    logger.info("PARAMETER" + "-" * 10)
    for attr, value in sorted(args.__dict__.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("---------" + "-" * 10)

    decomp_model = seq2seq_decomp_model(args)
    if args.action == 'train':
        decomp_model.train_model(args.epoch)

    elif args.action == 'eval':
        decomp_model.set_param()
        print(decomp_model.do_eval(decomp_model.dev_json))
        print(decomp_model.do_eval(decomp_model.test_json))
    elif args.action == 'ea':
        result = []
        decomp_model.set_param()
        js = decomp_model.test_json
        question_list = [x['question'] for x in js]  # 这个batch的question
        prefix_list = ['Decomp' for x in js]  # 这个batch的question
        predbatch , predbatch_no_decipher= decomp_model.batch_predict(question_list, prefix_list)  # 这个batch的预测有decipher的分解结果
        if args.seq_eval:
            result = [
                {'ID': js[index]['ID'], 'pred': predbatch[index], 'golden': js[index]['decomposition_2_part'],
                 'same': predbatch[index] == js[index]['decomposition_2_part'],
                 'no_decipher_same': predbatch_no_decipher[index] == js[index]['decomposition_2_part'],
                 } for index in range(len(js))]
        else:
            result = [
                {'ID': js[index]['ID'], 'pred': predbatch[index], 'golden': js[index]['decomposition'],
                 'same': predbatch[index] == js[index]['decomposition'],
                 'no_decipher_same': predbatch_no_decipher[index] == js[index]['decomposition'],
                 } for index in range(len(js))]
        print(len([x for x in result if x['same']])/len(result))
        print(len([x for x in result if x['no_decipher_same']])/len(result))
        savejson('../../data/ea_decomp/' + args.action + '_' + args.file_name + '_test', result)
    elif args.action == 'cwq_qa' or args.action == 'lc_qa':
        # predict file save path
        predict_file_name = '../../data/aaai_out/' + args.action + '_' + args.file_name

        logging.info('Predict decomposition file will saved in ' + predict_file_name)

        decomp_model.load_model()

        for part in ['test', 'dev', 'train']:
            js = readjson(
                '../../data/' + args.action.split('_')[0] + '/' + args.action.split('_')[0] + '_' + part + '.json')


            # load lc intent prediction
            if args.action == 'lc_qa':
                pre_type_file = readjson('../../data/pred_type/lc_pred_type_' + part + '.json')
            question_list = [clean_question(x['question']) for x in js]
            # prefix
            prefix_list = ['Decomp' for x in js]
            predbatch, predbatch_no_decipher = decomp_model.batch_predict(question_list, prefix_list)
            for index, item in enumerate(js):

                # predict compositionality type and intent
                if args.action == 'lc_qa':
                    pre_type_item = [x for x in pre_type_file if x['ID'] == js[index]['ID']][0]

                if args.action == 'lc_qa' and pre_type_item['compositionality_type'] == 'simple':
                    outer, inner = [js[index]['question']], []
                    qdt = outer_inner_list2qdt(outer, inner)
                    outer_no_decipher, inner_no_decipher = [js[index]['question']], []
                    qdt_no_decipher = outer_inner_list2qdt(outer_no_decipher, [])
                else:
                    outer, inner, qdt = linear_qdt_to_tree(predbatch[index], args.seq_eval)
                    outer_no_decipher, inner_no_decipher, qdt_no_decipher = linear_qdt_to_tree(
                        predbatch_no_decipher[index], args.seq_eval)
                js[index]['question'] = clean_question(js[index]['question'])
                # use decipher
                # js[index]['qdt'] = qdt
                js[index]['pred'] = predbatch[index]
                js[index]['subq1'] = outer
                js[index]['subq2'] = inner
                # without decipher
                # js[index]['qdt_no_decipher'] = qdt_no_decipher
                js[index]['pred_no_decipher'] = predbatch_no_decipher[index]
                js[index]['subq1_no_decipher'] = outer_no_decipher
                js[index]['subq2_no_decipher'] = inner_no_decipher

                # fill compositionality type
                js[index]['compositionality_type_golden'] = js[index]['compositionality_type']
                js[index]['compositionality_type'] = 'conjunction' if len(inner)==0 else 'composition'

                if args.action == 'lc_qa' and pre_type_item['compositionality_type'] == 'simple':
                    js[index]['compositionality_type'] = 'simple'


                # fill intent of lc, sample without seperator is set to simple
                if args.action == 'lc_qa':
                    js[index]['intent'] = pre_type_item['intent']
                    js[index]['intent_golden'] = pre_type_item['intent_golden']
                    if '[DES]' not in predbatch[index] and '[INQL]' not in predbatch[index] and '[INQR]' not in \
                            predbatch[index]:
                        js[index]['compositionality_type'] = 'simple'

                # delete useless property
                if args.action == 'cwq_qa':
                    for key in ['SExpr', 'decomposition', "constituency", "mask_entity", "entity"]:
                        del js[index][key]
                elif args.action == 'lc_qa':
                    for key in ['SExpr']:
                        del js[index][key]

            savejson(predict_file_name + '_' + part, js)
            logging.info('Predict file have saved in ' + predict_file_name + '_' + part + ' !')


if __name__ == "__main__":

    config = {'gpu_num': find_free_gpu(3, 20000),
              "max_length": 48,
              "batch_size": 64,
              'seed': 42,
              "epoch": 100,
              'other': '',
              'action': 'cwq_qa',
              'seq_eval': False,
              }

    config['file_name'] = 'decomp_seq2seq' +\
                          ('_seq' if config['seq_eval'] else '')+\
                          '_'+str(config['seed'])+\
                          ('_'+config['other'] if config['other']!='' else '')
    config['inference_batch_size'] = config['batch_size'] * 4

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--seq_eval", default=config['seq_eval'], type=bool)

    parser.add_argument("--max_length", default=config['max_length'], type=int)
    parser.add_argument("--batch_size", default=config['batch_size'], type=int)
    parser.add_argument("--inference_batch_size", default=config['inference_batch_size'], type=int)
    parser.add_argument("--epoch", default=config['epoch'], type=int)
    parser.add_argument("--gpu_num", default=config['gpu_num'], type=int)
    parser.add_argument("--seed", default=config['seed'], type=int)

    parser.add_argument("--other", default=config['other'],
                        type=str)
    parser.add_argument("--action", default=config['action'], type=str)
    parser.add_argument("--file_name", default=config['file_name'], type=str)

    args = parser.parse_args()
    main(args)

