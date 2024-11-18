
# Deteção de Embarcações de Pesca Ilegal Utilizando Imagens SAR


## Dependências

Instale as dependências usando conda:

```
cd DSA29_deep_learning_atividade/
conda env create -f environment.yml
```

## Pré-processamento

Primeiro, certifique-se de que as cenas de treinamento e validação foram extraídas para o mesmo diretório, por exemplo, `/xview3/all/images/`.  
Os rótulos de treinamento e validação devem ser concatenados e salvos em um arquivo CSV como `/xview3/all/labels.csv`.

Antes do treinamento, as grandes cenas precisam ser divididas em janelas (chips) de 800x800.  
Defina os caminhos e parâmetros em `data/configs/chipping_config.txt`, e então execute:

```
cd DSA29_deep_learning_atividade/src/
python -m xview3.processing.preprocessing ../data/configs/chipping_config.txt
```

## Treinamento Inicial

Aplicamos este modelo às cenas de treinamento do xView3 e incorporamos previsões de alta confiança como rótulos adicionais.  
Isso ocorre porque as cenas de treinamento do xView3 não estão completamente rotuladas, já que a maioria dos rótulos é derivada automaticamente de rastros do AIS.

Para treinar, configure os caminhos e parâmetros em `data/configs/initial.txt`, e então execute:

```
python -m xview3.training.train ../data/configs/initial.txt
```

Aplique o modelo treinado nas cenas de treinamento do xView3 e incorpore previsões de alta confiança como rótulos adicionais:

```
python -m xview3.infer.inference --image_folder /xview3/all/images/ --weights ../data/models/initial/best.pth --output out.csv --config_path ../data/configs/initial.txt --padding 400 --window_size 3072 --overlap 20 --scene_path ../data/splits/xview-train.txt
python -m xview3.eval.prune --in_path out.csv --out_path out-conf80.csv --conf 0.8
python -m xview3.misc.pred2label out-conf80.csv /xview3/all/chips/ out-conf80-tolabel.csv
python -m xview3.misc.pred2label_concat /xview3/all/chips/chip_annotations.csv out-conf80-tolabel.csv out-conf80-tolabel-concat.csv
python -m xview3.eval.prune --in_path out-conf80-tolabel-concat.csv --out_path out-conf80-tolabel-concat-prune.csv --nms 10
python -m xview3.misc.pred2label_fixlow out-conf80-tolabel-concat-prune.csv
python -m xview3.misc.pred2label_drop out-conf80-tolabel-concat-prune.csv out.csv out-conf80-tolabel-concat-prune-drop.csv
mv out-conf80-tolabel-concat-prune-drop.csv ../data/xval1b-conf80-concat-prune-drop.csv
```

## Treinamento Final

Agora podemos treinar o modelo final de detecção de objetos.  
Defina os caminhos e parâmetros em `data/configs/final.txt`, e então execute:

```
python -m xview3.training.train ../data/configs/final.txt
```

## Predição de Atributos

Usamos um modelo separado para prever os atributos `is_vessel`, `is_fishing` e o comprimento da embarcação.

```
python -m xview3.postprocess.v2.make_csv ./xview3/all/chips/chip_annotations.csv out.csv ../data/splits/our-train.txt ./xview3/postprocess/labels.csv
python -m xview3.postprocess.v2.get_boxes ./xview3/postprocess/labels.csv ./xview3/all/chips/ ./xview3/postprocess/boxes/
python -m xview3.postprocess.v2.train ./xview3/postprocess/model.pth ./xview3/postprocess/labels.csv ./xview3/postprocess/boxes/
```

## Inferência

Suponha que as imagens de teste estão em um diretório como `/xview3/test/images/`. Primeiro, aplique o detector de objetos:

```
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20
python -m xview3.eval.prune --in_path out.csv --out_path out-prune.csv --nms 10
```

Agora aplique o modelo de predição de atributos:

```
python -m xview3.postprocess.v2.infer /xview3/postprocess/model.pth out-prune.csv /xview3/test/chips/ out-prune-attribute.csv attribute
```

## Aumento de Dados em Teste

Empregamos aumento de dados durante o teste na nossa submissão final, o que proporciona uma pequena melhoria de 0,5% no desempenho

```
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-1.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-2.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --fliplr True
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-3.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --flipud True
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-4.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --fliplr True --flipud True
python -m xview3.eval.ensemble out-1.csv out-2.csv out-3.csv out-4.csv out-tta.csv
python -m xview3.eval.prune --in_path out-tta.csv --out_path out-tta-prune.csv --nms 10
python -m xview3.postprocess.v2.infer /xview3/postprocess/model.pth out-tta-prune.csv /xview3/test/chips/ out-tta-prune-attribute.csv attribute
```

## Limite de Confiança

Ajustamos o limite de confiança no conjunto de validação. Repita os passos de inferência com aumento de dados no conjunto `our-validation.txt` para obter `out-validation-tta-prune-attribute.csv`. Então:

```
python -m xview3.eval.metric --label_file /xview3/all/chips/chip_annotations.csv --scene_path ../data/splits/our-validation.txt --costly_dist --drop_low_detect --inference_file out-validation-tta-prune-attribute.csv --threshold -1
python -m xview3.eval.prune --in_path out-tta-prune-attribute.csv --out_path submit.csv --conf 0.3 # Alterar para o melhor limite de confiança.
```
