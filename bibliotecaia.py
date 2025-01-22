import numpy as np
from PIL import Image
import numpy as np
import keyboard
import os
from time import sleep

class modelClassificate1:
    def __init__(self):
        print('Bem vindo a biblioteca IA para multiclassificação\nAinda em desenvolvimento!','\nUse o help() para ver as funcionalidades da biblioteca.')

    def help(self):

        color_red = "\033[91m"      # Vermelho
        color_green = "\033[92m"    # Verde
        color_yellow = "\033[93m"   # Amarelo
        color_blue = "\033[94m"     # Azul
        color_reset = "\033[0m"
        count = 1
        space = False
        print('Pressione "up" ou "down" para navegar, e "space" para selecionar.')

        while True:
            # Detecção de teclas "up" e "down"
            if keyboard.is_pressed('up'):
                count -= 1
                sleep(0.15)  # Pequena pausa para evitar múltiplas detecções instantâneas
            elif keyboard.is_pressed('down'):
                count += 1
                sleep(0.15)

            # Limites para o contador (menu cíclico)
            if count < 1:
                count = 3
            elif count > 3:
                count = 1

            # Itens do menu
            menu_items = [
                '----------Menu----------',
                '1 - Funções matemáticas   (_) ',
                '2 - Leitura de imagens    (_) ',
                '3 - Normalização          (_) '
            ]

            # Atualiza a seleção no menu
            for i in range(1, len(menu_items)):
                menu_items[i] = menu_items[i].replace('X', '_')  # Limpa a marcação antiga
                if i == count:
                    x=f'{color_blue}X{color_reset}'
                    menu_items[i] = menu_items[i].replace('_', x )  # Marcação da nova seleção

            # Exibe o menu atualizado
            os.system('cls' if os.name == 'nt' else 'clear')  # Limpa o terminal
            for item in menu_items:
                print(item)

            # Detecção de tecla "space" para seleção
            if keyboard.is_pressed('space'):
                space = True
                os.system('cls' if os.name == 'nt' else 'clear')
                if count == 1:
                    print(f'{color_red}----- Funções Matemáticas -----{color_reset}')
                    print(f'{color_green} # 1. Função para inicializar os pesos{color_reset}: {color_blue}initialize_weights(){color_reset}')
                    print('  - Descrição: Inicializa os pesos aleatórios da rede.')
                    print('  - Parâmetros: \n    - indv_entry_data (np.array): Dados de entrada.\n    - indv_goals_data (np.array): Dados de saída.\n')
                    
                    print(f'{color_green}# 2. Função de ativação softmax:{color_reset}{color_blue} softmax(){color_reset}')
                    print('  - Descrição: Função de ativação que transforma os valores em probabilidades.')
                    print('  - Parâmetros: \n    - z (np.array): Multiplicação das entradas pelos pesos.\n')
                    
                    print(f'{color_green}# 3. Função para calcular perda:{color_reset}{color_blue} cross_entopry_loss(){color_reset}')
                    print('  - Descrição: Calcula a perda entre o valor real e o previsto usando a entropia cruzada.')
                    print('  - Parâmetros: \n    - y_true (np.array): Valores reais.\n    - y_pred (np.array): Valores previstos.\n')
                    
                    print(f'{color_green}# 4. Função para treinar o modelo:{color_reset}{color_blue} train_model(){color_reset}')
                    print('  - Descrição: Ajusta os pesos da rede com base no erro, utilizando gradient descent.')
                    print('  - Parâmetros: \n    - entry (np.array): Dados de entrada.\n    - goals (np.array): Dados esperados.\n    - weights (np.array): Pesos do modelo.\n    - alpha (float): Taxa de aprendizado.\n    - epochs (int): Número de épocas de treinamento.\n    - show_loss (bool): Exibe a perda durante o treinamento se True.\n')

                elif count == 2:
                    print(f'{color_red}----- Leitura de Imagens -----{color_reset}')
                    print(f'{color_green}# 1. Função para leitura de imagem:{color_reset}{color_blue} read_image(){color_reset}')
                    print('  - Descrição: Lê e converte uma imagem para um array de pixels.')
                    print('  - Parâmetros: \n    - image_path (str): Caminho da imagem.\n    - normalize_image (bool): Se True, normaliza os valores dos pixels.\n    - simple_normalize (bool): Se True, normaliza os pixels para 0 ou 1.\n')
                    
                    print(f'{color_green}# 2. Função para mostrar detalhes da imagem:`{color_reset}{color_blue} show_details_image(){color_reset}')
                    print('  - Descrição: Exibe detalhes da imagem, como modo e tamanho.')
                    print('  - Parâmetros: \n    - image_path (str): Caminho da imagem.\n')
                    
                    print(f'{color_green}# 3. Função para converter imagem em array de pixels:{color_reset}{color_blue} convert_image(){color_reset}')
                    print('  - Descrição: Converte uma lista de imagens em arrays de pixels achatados.')
                    print('  - Parâmetros: \n    - _list_ (list): Lista com os nomes das imagens.\n    - pasta (str): Nome da pasta onde as imagens estão.\n')

                elif count == 3:
                    print(f'{color_red}----- Normalização -----{color_reset}')
                    print(f'{color_green}# 1. Função de normalização simples:{color_reset}{color_blue} simple_normalize_image(){color_reset}')
                    print('  - Descrição: Normaliza os pixels da imagem, transformando os valores 255 em 0 e o restante em 1.')
                    print('  - Parâmetros: \n    - image_data_array (np.array): Array com os dados dos pixels da imagem.\n')

                input('Pressione Enter para voltar ao menu.')
                os.system('cls' if os.name == 'nt' else 'clear')

            sleep(0.1)
            
    def initialize_weights(self,indv_entry_data,indv_goals_data):
        """
        Inicializa os pesos da rede com valores aleatórios.

        Args:
            indv_entry_data (np.array): Dados de entrada para o modelo.
            indv_goals_data (np.array): Dados esperados para o modelo.

        Returns:
            np.array: Pesos inicializados aleatoriamente.
        """
        w_ights=np.random.rand(indv_entry_data.shape[1],indv_goals_data.shape[0])
        return w_ights
    
    def softmax(self,z):
        exp_z=np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entopry_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15))/y_true.shape[0]
    
    def train_model(self, entry, goals, weights, alpha=0.01,epochs=1000, show_loss=False):
        """
            Treina o modelo ajustando os pesos com base no erro.

            Args:
                entry (np.array): Dados de entrada.
                goals (np.array): Dados esperados.
                weights (np.array): Pesos do modelo.
                alpha (float): Taxa de aprendizado.
                epochs (int): Número de épocas de treinamento.
                show_loss (bool): Exibe a perda durante o treinamento se True.

            Returns:
                np.array: Pesos ajustados após o treinamento.
        """
        m = entry.shape[0]

        for epoch in range(epochs):

            z=np.dot(entry,weights)
            y_pred=self.softmax(z)

            loss=self.cross_entopry_loss(goals,y_pred)

            gradient = np.dot(entry.T, (y_pred - goals)) / m

            weights-=alpha * gradient

            if show_loss:
                if epoch % 100 == 0:
                    print(f'Epoch {epoch} ,loss: {loss}')
        return weights
    
    def read_image(self, image_path, normalize_image = True, simple_normalize=True):
        """
            Lê e converte uma imagem para um array de pixels.

            Args:
                image_path (str): Caminho da imagem.
                normalize_image (bool): Se True, normaliza os valores dos pixels.
                simple_normalize (bool): Se True, normaliza os pixels para 0 ou 1.

            Returns:
                np.array: Array com os valores dos pixels.
        """
        image=Image.open(image_path)

        image=image.convert('L')
        
        pixel_data=list(image.getdata())

        pixel_array=np.array(pixel_data)

        if normalize_image:
            if simple_normalize:
                return np.array(self.simple_normalize_image(pixel_data))
            else:
                
                return np.array(self.advanced_normalize_image(pixel_data))
        else:
            return pixel_array  
    
    def simple_normalize_image(self,image_data_array):
        """
            Normaliza os pixels da imagem, transformando os valores 255 em 0 e o restante em 1.

            Args:
                image_data_array (np.array): Array com os dados dos pixels da imagem.

            Returns:
                list: Lista com os valores normalizados.
        """
        lista=[]

        for item in image_data_array:
            if item == 255:
                lista.append(0)
            else:
                lista.append(1)
        return lista
    
    def flattern_array(self,array):
        return array.flatten()
    
    def show_details_image(self,image_path):
        image=Image.open(image_path)
        print(f'Imagem carregada: {image_path}')
        print(f'modo da imagem: {image.mode}')
        print(f'Tamanho da imagem: {image.size}')

    def predict(self, entrada,modelo):
        return np.argmax(self.softmax(np.dot(entrada,modelo)),axis=1)
    
    def calculate_goals(self, entry, goals):
        """
        Calcula a acurácia das previsões em relação aos valores reais.

        Args:
            entry (np.array): Previsões do modelo.
            goals (np.array): Valores reais.

        Returns:
            float: Acurácia das previsões.
        """
        correct_predictions = np.sum(entry == goals)
        accuracy = correct_predictions / len(goals)
        return accuracy
    
    def convert_image(self,_list_,pasta='treino',normalize_i=True,advanced_normalize=False):
        lista=[]

        for nome in _list_:
            image=self.read_image(f'./{pasta}/{nome}',normalize_image=normalize_i,simple_normalize=advanced_normalize)
        
            flat=self.flattern_array(image)


            lista.append(flat)
        return np.array(lista)
    
    def eye_gab(self,lista_de_respostas):
        lista_de_respostas_eye=np.eye(lista_de_respostas.shape[0])[lista_de_respostas]
        return lista_de_respostas_eye
    
    def final_test_pred(self,images_for_com,nome_pasta,model_trained,classes):
        lista_imagens_comprimidas=self.convert_image(images_for_com,nome_pasta)
        pred=self.predict(lista_imagens_comprimidas,model_trained)
        for item in pred:
            print(f'{classes[item]}')
        return pred

    def advanced_normalize_image(self, image_data_array):
        """
        Normaliza os pixels da imagem para um intervalo de 0 a 1, considerando múltiplas cores.

        Args:
            image_data_array (np.array): Array com os dados dos pixels da imagem.

        Returns:
            list: Lista com os valores normalizados.
        """
        lista = []
        max_value = np.max(image_data_array)
        min_value = np.min(image_data_array)
        
        for item in image_data_array:
            normalized_value = (item - min_value) / (max_value - min_value)
            lista.append(normalized_value)
    
        
        return lista

