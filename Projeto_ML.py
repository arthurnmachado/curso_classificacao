#!/usr/bin/env python
# coding: utf-8

# # 1.2 - Importando os dados
# 
# 
# 

# In[1]:


import pandas as pd


# In[5]:


dados = pd.read_csv('https://drive.google.com/uc?export=download&id=1pILmRp-OU-SzZnLvEot3uywVpqqAsbW0')


# In[6]:


dados.shape


# In[7]:


dados.head()


# # 1.3 - Diferentes Variáveis

# In[11]:


# modificação de forma manual
traducao_dic = {
    'Sim': 1,
    'Nao': 0
}
dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)
dadosmodificados.head()


# In[15]:


# transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                                         axis=1))

# junção dos dados transformados com os que já tinhamos
dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)


# In[16]:


dados_final.head()


# In[17]:


dados_final.shape


# # 1.5 - Definição Formal

# Informações para classificação:
# 
# 𝒳 = inputs
# 
# 𝒴 = outputs
# 
# 

# In[19]:


pd.set_option('display.max_columns', 39)


# In[20]:


dados_final.head()


# 𝓨ᵢ = 𝒇(𝒳ᵢ)

# In[21]:


Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]


# In[22]:


# Ymaria = ?


# # Balanceando dados

# In[23]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ax = sns.countplot(x='Churn', data=dados_final)


# In[24]:


get_ipython().system('pip install -U imbalanced-learn')


# In[25]:


# Para podermos aplicar o SMOTE, devemos separar  os dados em variáveis características e resposta

X = dados_final.drop('Churn', axis = 1)
y = dados_final['Churn']


# In[26]:


from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state=123)  # Instancia um objeto da classe SMOTE
X, y = smt.fit_resample(X, y)  # Realiza a reamostragem do conjunto de dados


# In[27]:


dados_final = pd.concat([X, y], axis=1)  # Concatena a variável target (y) com as features (X)

# Verifica se o balanceamento e a concatenação estão corretos.
dados_final.head(2)


# In[28]:


ax = sns.countplot(x='Churn', data=dados_final)  # plotando a variável target balanceada.


# In[ ]:




