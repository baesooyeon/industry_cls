import torch
import torch.nn as nn
import torch.nn.functional as F

def get_classifier(input_size, num_classes, hidden_size, bias, dr_rate, num_layers, batchnorm, layernorm):
    classifier=[]
    if num_layers == 1:
        if dr_rate:
            classifier.append(nn.Dropout(p=dr_rate))
        classifier.append(nn.Linear(input_size, num_classes, bias=bias))
    else:
        for i in range(num_layers):
            """
            (dropout)
            linear(768, hidden_size)
            (normalization)
            activation
                |
            (dropout)
            linear(hidden_size, hidden_size)
            (normalization)
            activation
                |
               ...
                |
            (dropout)
            linear((hidden_size, num_classes)
            """
            # drop out
            if dr_rate:
                classifier.append(nn.Dropout(p=dr_rate))

            # linear layer
            if i == 0: # 첫번째 층
                classifier.append(nn.Linear(input_size, hidden_size, bias=True))
            elif i != num_layers-1: # 중간 층
                classifier.append(nn.Linear(hidden_size, hidden_size, bias=True))
            else: # 마지막 층
                classifier.append(nn.Linear(hidden_size, num_classes, bias=True))

            # normalization
            if i != num_layers-1: # 마지막 층이 아니면
                if batchnorm:
                    classifier.append(nn.BatchNorm1d(hidden_size))
                if layernorm:
                    classifier.append(nn.LayerNorm(hidden_size))

            # activation
            if i != num_layers-1: # 마지막 층이 아니면
                classifier.append(nn.ReLU())
    return nn.Sequential(*classifier)

class _LMClassifier(nn.Module):
    def __init__(self,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(_LMClassifier, self).__init__()
        assert not all([batchnorm, layernorm]), 'use one normalization among batchnorm and layernorm. now got both of them.' 
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.linear_input_size = linear_input_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.freeze = freeze
        self.dr_rate = dr_rate
        self.bias = bias
        
        # lm_head : classifier
        self.classifier = get_classifier(linear_input_size, num_classes, hidden_size, bias, dr_rate, num_layers, batchnorm, layernorm)

        
class KobertClassifier(_LMClassifier):
    def __init__(self, bert,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size = 4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(KobertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bert = bert
#         if self.freeze:
#             for child in self.bert.children():
#                 for param in child.parameters():
#                     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooler = self.bert(input_ids=input_ids.long(),
                              token_type_ids=token_type_ids.long(),
                              attention_mask=attention_mask.float())
        return self.classifier(pooler)
    
class BertClassifier(_LMClassifier):
    def __init__(self, bert,
                 num_classes, num_layers=1, linear_input_size=768, hidden_size = 4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(BertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bert = bert
#         if self.freeze:
#             for child in self.bert.children():
#                 for param in child.parameters():
#                     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        dec_output = self.bert(input_ids=input_ids.long(),
                               token_type_ids=token_type_ids.long(),
                               attention_mask=attention_mask.float())
        return self.classifier(dec_output.pooler_output)
        

class KogptClassifier(_LMClassifier):
    """ kogpt2 or kogpt3 """
    def __init__(self, kogpt, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(KogptClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.kogpt = kogpt
#         if self.freeze:
#             for child in self.kogpt.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.kogpt.transformer(input_ids=input_ids, 
                                  token_type_ids=token_type_ids,
                                  attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class ElectraClassifier(_LMClassifier):
    def __init__(self, electra, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(ElectraClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.electra = electra
#         if self.freeze:
#             for child in self.electra.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.electra(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class AlbertClassifier(_LMClassifier):
    def __init__(self, albert, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(AlbertClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.albert = albert
#         if self.freeze:
#             for child in self.albert.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.albert(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class FunnelClassifier(_LMClassifier):
    def __init__(self, funnel, num_classes, num_layers=1, linear_input_size=768, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(FunnelClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.funnel = funnel
#         if self.freeze:
#             for child in self.funnel.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_output = self.funnel(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
    
class BartClassifier(_LMClassifier):
    def __init__(self, bart, num_classes, num_layers=1, linear_input_size=1024, hidden_size=4026, activation='relu', dr_rate=None, bias=True,
                 batchnorm=False, layernorm=False,
                 freeze=True):
        super(BartClassifier, self).__init__(num_classes, num_layers, linear_input_size, hidden_size, activation, dr_rate, bias, batchnorm, layernorm, freeze)
        self.bart = bart
#         if self.freeze:
#             for child in self.bart.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask):
        # transformer decoder output
        # size : (b, n_dec_seq, n_hidden)
        dec_outputs = self.bart(input_ids=input_ids, 
                               attention_mask = attention_mask)

        # classifier output
        # size : (b, n_hidden)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        # size : (b, num_classes)
        logits_cls = self.classifier(dec_outputs)
        return logits_cls
        
class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes,
#                  kobert=None, mlbert=None, bert=None, albert=None, 
#                  kobart=None, asbart=None,
#                  kogpt2=None, kogpt3=None, electra=None, funnel=None,
                 kobert_classifiers=[], mlbert_classifiers=[], bert_classifiers=[], albert_classifiers=[],
                 kobart_classifiers=[], asbart_classifiers=[],
                 kogpt2_classifiers=[], kogpt3_classifiers=[], electra_classifiers=[], funnel_classifiers=[]):
        super(EnsembleClassifier, self).__init__()
#         assert (kobert is not None)==(len(kobert_classifiers)!=0),  'expect both of kobert and kobert_classifiers, but get one of them'
#         assert (kogpt2 is not None)==(len(kogpt2_classifiers)!=0),  'expect both of kogpt2 and kogpt2_classifiers, but get one of them'
#         assert (mlbert is not None)==(len(mlbert_classifiers)!=0),  'expect both of mlbert and mlbert_classifiers, but get one of them'
#         assert (hanbert is not None)==(len(hanbert_classifiers)!=0),'expect both of hanbert and hanbert_classifiers, but get one of them'
#         assert (dstbert is not None)==(len(dstbert_classifiers)!=0),'expect both of dstbert and dstbert_classifiers, but get one of them'
#         assert (skobert is not None)==(len(skobert_classifiers)!=0),'expect both of skobert and skobert_classifiers, but get one of them'
        self.num_classes = num_classes
        self.kobert_classifiers = kobert_classifiers
        self.mlbert_classifiers = mlbert_classifiers
        self.bert_classifiers = bert_classifiers
        self.albert_classifiers = albert_classifiers
        self.asbart_classifiers = asbart_classifiers
        self.asbart_classifiers = asbart_classifiers
        self.kogpt2_classifiers = kogpt2_classifiers
        self.kogpt3_classifiers = kogpt3_classifiers
        self.electra_classifiers = electra_classifiers
        self.funnel_classifiers = funnel_classifiers
        
#         self.kobert, self.kobert_classifiers = kobert, kobert_classifiers
#         self.mlbert, self.mlbert_classifiers = mlbert, mlbert_classifiers
#         self.bert, self.bert_classifiers = bert, bert_classifiers
#         self.albert, self.albert_classifiers = albert, albert_classifiers
#         self.kobart, self.asbart_classifiers = kobart, asbart_classifiers
#         self.asbart, self.asbart_classifiers = asbart, asbart_classifiers
#         self.kogpt2, self.kogpt2_classifiers = kogpt2, kogpt2_classifiers
#         self.kogpt3, self.kogpt3_classifiers = kogpt3, kogpt3_classifiers
#         self.electra, self.electra_classifiers = electra, electra_classifiers
#         self.funnel, self.funnel_classifiers = funnel, funnel_classifiers
        
#         for backbone in [self.kobert, self.mlbert, self.bert, self.albert,
#                          self.kobart, self.asbart,
#                          self.kogpt2, self.kogpt3, self.electra, self.funnel,
# #                          self.mlbert, self.hanbert, self.dstbert, self.skobert
#                         ]:
#             if backbone is not None:
#                 self.freeze(backbone)
                
#     def freeze(self, backbone):
#         for child in backbone.children():
#             for param in child.parameters():
#                 param.requires_grad = False

    def forward(self,
                kobert=None, mlbert=None, bert=None, albert=None,
                kobart=None, asbart=None,
                kogpt2=None, kogpt3=None, electra=None, funnel=None):
        output = []
        # bert output
        if self.kobert:
            output += self.kobert_forward(kobert[:,0].long(),  kobert[:,1].float(), kobert[:,2].long())
        if self.mlbert:
            output += self.mlbert_forward(mlbert[:,0], mlbert[:,1], mlbert[:,2])
        if self.bert:
            output += self.bert_forward(bert[:, 0], bert[:, 1])
        if self.albert:
            output += self.albert_forward(albert[:,0], albert[:,1], albert[:,2])
        if self.kobart:
            output += self.kobart_forward(kobart[:,0], asbart[:,1], asbart[:,2])
        if self.asbart:
            output += self.asbart_forward(asbart[:,0], asbart[:,1], asbart[:,2])
        if self.kogpt2:
            output += self.kogpt2_forward(kogpt2[:,0], kogpt2[:,1], kogpt2[:,2])
        if self.kogpt3:
            output += self.kogpt3_forward(kogpt3[:,0], kogpt3[:,1], kogpt3[:,2])
        if self.electra:
            output += self.electra_forward(electra[:,0], electra[:,1], electra[:,2])
        if self.funnel:
            output += self.funnel_forward(funnel[:,0], funnel[:,1], funnel[:,2])

        output = torch.stack(output)
        return output
    
    def kobert_forward(self, input_ids, attention_mask, token_type_ids):
#         _, pooler = self.kobert(input_ids=input_ids,
#                                 attention_mask=attention_mask,
#                                 token_type_ids=token_type_ids)
        return [F.softmax(classifier(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids), dim=1) for classifier in self.kobert_classifiers]
    
    def mlbert_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.mlbert(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        return [F.softmax(classifier(dec_outputs.pooler_output), dim=1) for classifier in self.mlbert_classifiers]
    
    def bert_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.bert(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        return [F.softmax(classifier(dec_outputs.pooler_output), dim=1) for classifier in self.bert_classifiers]
    
    def albert_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.albert(input_ids=input_ids, 
                                 token_type_ids=token_type_ids,
                                 attention_mask = attention_mask)
        dec_outputs = dec_output.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.albert_classifiers]
    
    def kobart_forward(self, input_ids, attention_mask):
        dec_outputs = self.kobart(input_ids=input_ids, 
                                 attention_mask = attention_mask)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.kobart_classifiers]
    
    def asbart_forward(self, input_ids, attention_mask):
        dec_outputs = self.asbart(input_ids=input_ids, 
                                      attention_mask = attention_mask)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.asbart_classifiers]
    
    def kogpt2_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.kogpt2.transformer(input_ids=input_ids,
                                             attention_mask = attention_mask,
                                             token_type_ids=token_type_ids)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.kogpt2_classifiers]
    
    def kogpt3_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.kogpt3.transformer(input_ids=input_ids,
                                             attention_mask = attention_mask,
                                             token_type_ids=token_type_ids)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.kogpt3_classifiers]
    
    def electra_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.electra(input_ids=input_ids,
                                  attention_mask = attention_mask,
                                  token_type_ids=token_type_ids)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.electra_classifiers]
    
    def funnel_forward(self, input_ids, attention_mask, token_type_ids):
        dec_outputs = self.funnel(input_ids=input_ids,
                                 attention_mask = attention_mask,
                                 token_type_ids=token_type_ids)
        dec_outputs = dec_outputs.last_hidden_state[:, -1].contiguous() # 마지막 예측 토큰을 분류값으로 사용
        return [F.softmax(classifier(dec_outputs), dim=1) for classifier in self.funnel_classifiers]