from typing import List, Tuple, Callable, Optional
from typing import Tuple, Dict

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

import numpy as np
import random
import secrets
import time
import math

# Import de GAConfig (mesmo diretório)
from config import GAConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


N_QUBITS = 8


class Individuo:
    def __init__(self,  geracao:        int,   
                        num_portas:     int,                    
                        num_variaveis:  int,
                        door_type:      int, 
                        domain:         List[Tuple[float, float]],  
                        seed:           int,
                        sigma_es:       float = 1.0,
                        annealing_window: bool = False,
                        sigma_es_ativo: bool = False):
    

        # apenas passando adiante os parâmetros
        self.geracao            = geracao
        self.max_num_portas     = num_portas
        self.num_portas         = num_portas    #random.randint(1, num_portas)
        self.num_variaveis      = num_variaveis
        self.num_qubits         = num_variaveis * N_QUBITS
        self.door_type          = door_type
        self.domain             = domain if domain else [(-1.0, 1.0) for _ in range(num_variaveis)]
        self.seed               = seed        
        self.fitness            = None
        self.sigma_es           = sigma_es
        self.sigma_es_ativo     = sigma_es_ativo
        self.annealing_window   = annealing_window
        self.location           = None  # Cache de localização (media_valores)
        self.simulation_results = None  # Cache de resultados da simulação

        
        # determinando universo de portas
        if(self.door_type == -1):   # Classical pure (single-qubit only, no entanglement)

            self.PORTAS1Q = ['X','I']
            self.PORTAS2Q = []
            self.PORTAS3Q = []

        elif(self.door_type == 0):    # Classic

            self.PORTAS1Q = ['X','I']
            self.PORTAS2Q = ['CNOT','SWAP']
            self.PORTAS3Q = ['TOFF','FREAD']

        elif(self.door_type == 1):  # Clifford

            self.PORTAS1Q = ['X','I','Y', 'Z', 'S','Sd', 'H'] 
            self.PORTAS2Q = ['CNOT','SWAP']
            self.PORTAS3Q = []

        elif(self.door_type == 2):  # General / Hilbert 

            self.PORTAS1Q = ['X','I','Y', 'Z', 'S','Sd', 'H','T','Td']
            self.PORTAS2Q = ['CNOT','SWAP']
            self.PORTAS3Q = ['TOFF', 'FREAD']

        elif(self.door_type == 3):  # Quantum without entanglement (single-qubit only)

            self.PORTAS1Q = ['X','I','Y', 'Z', 'S','Sd', 'H','T','Td']
            self.PORTAS2Q = []
            self.PORTAS3Q = []

        else:
            print(f"Error: invalid set of doors indicator {self.door_type}")
            
        self.PORTAS = self.PORTAS1Q + self.PORTAS2Q + self.PORTAS3Q
        
        # gerando genes aleatórios
        self.genoma = self._gerar_genes_aleatorios()
        

    @classmethod
    def from_config(cls, config, geracao: int = 0, seed: int = 0) -> 'Individuo':
        """
        Cria um Individuo a partir de um GAConfig.
        
        Uso:
            from config import GAConfig
            config = GAConfig(num_variaveis=2, domain=[(-5,5),(-1,1)], sigma_es=True)
            ind = Individuo.from_config(config, geracao=0, seed=42)
        """
        sigma = config.sigma_es_initial if config.sigma_es else 1.0
        return cls(
            geracao=geracao,
            num_portas=config.num_portas,
            num_variaveis=config.num_variaveis,
            door_type=config.door_type,
            domain=config.domain,
            seed=seed,
            sigma_es=sigma,
            annealing_window=config.annealing_window,
            sigma_es_ativo=config.sigma_es
        )

    
    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # início da parte de métodos


    def _gerar_genes_aleatorios(self) -> np.ndarray:
        
        genes = np.array([[('K', None, None, None) for _ in range(self.num_portas)] for q in range(self.num_qubits)], dtype=object)
        rng = random.Random(self.seed)

        # Definir portas disponíveis
        portas_disponiveis = self.PORTAS
        
        #[print(q) for q in range(self.num_qubits)]  
        disponiveis = [q for q in range(self.num_qubits)]  

        # Tratamento especial para a primeira coluna quando p_h > 0
        if self.geracao == 0 and self.door_type > 0:

            # Sorteia quantos Hs serão colocados (entre 1 e num_qubits)
            num_h = rng.randint(1, self.num_qubits)
            #print("nH: ", num_h, flush=True)

            # Sorteia quais qubits receberão H
            qubits_com_h = rng.sample(disponiveis, num_h)

            # Aplica H nos qubits sorteados
            for q in qubits_com_h:
                genes[q, 0] = ('H', None, None, q)
                if q in disponiveis:
                    disponiveis.remove(q)


        for i in range(self.num_portas):

            # Lista de qubits disponíveis nesta coluna (índices)
            disponiveis = [q for q in range(self.num_qubits) if genes[q, i][0] == 'K']

            while disponiveis:
                q = disponiveis.pop(0)  # Pega o próximo qubit disponível

                # Determinar portas possíveis baseado nos qubits restantes
                possiveis_portas = self.PORTAS1Q.copy()
                qubits_restantes = [x for x in disponiveis if x > q]  # Qubits após o atual

                # Verificar possibilidade de portas de 2 qubits
                if len(qubits_restantes) >= 1 and q <= self.num_qubits - 2:
                    possiveis_portas += self.PORTAS2Q

                # Verificar possibilidade de portas de 3 qubits
                if len(qubits_restantes) >= 2 and q <= self.num_qubits - 3:
                    possiveis_portas += self.PORTAS3Q

                # Escolher porta aleatória dentre as possíveis
                op = rng.choice(possiveis_portas)


                # Aplicar a porta selecionada
                if op in self.PORTAS1Q:
                    genes[q, i] = (op, None, None, q)


                elif op in self.PORTAS2Q and len(qubits_restantes) >= 1:
                    alvo = rng.choice(qubits_restantes)

                    controle = q
                    if rng.random() < 0.5:  # 50% de chance de inverter os papéis
                        controle, alvo = alvo, q  # Inverte os papéis

                    # Registra a operação nos genes
                    genes[controle, i] = (op, controle, None, alvo)
                    genes[alvo, i] = (op, controle, None, alvo)
                    # Remove o alvo dos disponíveis
                    if alvo in disponiveis:
                        disponiveis.remove(alvo)


                elif op in self.PORTAS3Q and len(qubits_restantes) >= 2:                   
                    qubits_operacao = [q] + rng.sample(qubits_restantes, 2)

                    # Sorteia quais serão os 2 controles e qual será o alvo
                    rng.shuffle(qubits_operacao)  # Embaralha os papéis
                    controle1, controle2, alvo = qubits_operacao  # Desempacota

                    # Registra a operação nos genes de todos os 3 qubits
                    for qubit in [controle1, controle2, alvo]:
                        genes[qubit, i] = (op, controle1, controle2, alvo)
                        # Remove os alvos dos disponíveis
                        if qubit in disponiveis:
                            disponiveis.remove(qubit)
        """
        linhas = [f""]
        for q in range(self.num_qubits):
            linha = f"q{q}: " + ' | '.join(
                f"{str(genes[q, i])}" for i in range(self.num_portas)
            )
            linhas.append(linha)
        print('\n'.join(linhas))
        """

        return genes
    

    def get_depth(self) -> int:        
        return self.num_portas


    def construir_circuito_qiskit(self, measure: bool = True) -> QuantumCircuit:
        """
        Constrói o circuito quântico baseado nos genes do indivíduo.
        Retorna apenas o circuito (sem simulação).

        Args:
            measure: Se True (default), adiciona as medições no final.
                     Se False, retorna o circuito sem medição — necessário para
                     extrair o statevector exato (a medição colapsaria o estado).
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        rng = random.Random(self.seed)
    
        for q in range(self.num_qubits):
            if rng.random() < 0.5:
                qc.x(q)
        qc.barrier()
    
        # Aplicação dos genes (portas quânticas)
        for i in range(self.num_portas):
            self._apply_gate(qc, i)
    
        qc.barrier()
        if measure:
            qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc


    def simular_circuito(self, qc: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """
        Simula o circuito quântico dado e retorna o dicionário de contagens das medições.
        """
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        resultados = job.result().get_counts(qc)
        return resultados


    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # cálculo de fitness e operações de valores bits → decimais


    def calcular_fitness(self, fitness_function: Callable[[List[float]], float], 
                         shots: int = 1024, dataCollection: bool = False,
                         exact: bool = False) -> float:

        """
        Simula o circuito com a quantidade de shots definida.
        Converte cada resultado de medição em valores reais,
        calcula a média ponderada dos valores reais e aplica a função de fitness ao final.
        
        Args:
            fitness_function: Função de fitness a ser aplicada.
            shots: Número de shots da simulação.
            dataCollection: Se True, salva localização e resultados no cache.
            exact: Se True, avalia via statevector exato (𝔼[x] sem ruído de amostragem).
        """

        if exact:
            return self._calcular_fitness_exato(fitness_function, dataCollection=dataCollection)

        qc = self.construir_circuito_qiskit()
        resultados = self.simular_circuito(qc, shots=shots)
    
        total_shots = sum(resultados.values())
        if total_shots == 0:
            self.fitness = 0.0
            return self.fitness
    
        soma_valores = [0.0 for _ in range(self.num_variaveis)]
    
        for estado, count in resultados.items():
            valores = self._converter_estado_para_reais(estado[::-1])
            for i in range(self.num_variaveis):
                soma_valores[i] += valores[i] * count
    
        media_valores = [s / total_shots for s in soma_valores]
        
        # Salvar cache se dataCollection habilitado
        if dataCollection:
            self.location = media_valores
            self.simulation_results = resultados
    
        self.fitness = fitness_function(media_valores)
        return self.fitness


    def _calcular_fitness_exato(self, fitness_function: Callable[[List[float]], float],
                                dataCollection: bool = False) -> float:
        """
        Avalia o fitness via statevector exato.

        Em vez de amostrar Nshots vezes e aproximar 𝔼[x] pela média amostral,
        extrai o statevector |ψ⟩ do circuito e calcula a esperança exata:

            x* = Σ_z |ψ(z)|² · x(z)

        O paper define o fitness como f(𝔼_{z∼P_C}[x(z)]); a média de shots é
        apenas uma estimativa ruidosa desse valor. O statevector entrega o
        próprio 𝔼[x], sem ruído de amostragem e sem o custo de 1024 shots.
        """
        qc = self.construir_circuito_qiskit(measure=False)
        qc.save_statevector()

        simulator = AerSimulator()
        job = simulator.run(qc, shots=1)
        data = job.result().data()
        probs = np.abs(data['statevector']) ** 2

        # Estado como tensor 2×...×2 (um eixo por qubit).
        # reshape em ordem C põe o eixo 0 = MSB (qubit num_qubits-1); transpor
        # para que o eixo k corresponda ao qubit k (convenção do Qiskit:
        # qubit 0 = bit menos significativo do índice do statevector).
        amps = probs.reshape([2] * self.num_qubits)
        amps = amps.transpose(tuple(reversed(range(self.num_qubits))))

        max_int = 2 ** N_QUBITS - 1
        media_valores = [0.0 for _ in range(self.num_variaveis)]

        for var in range(self.num_variaveis):
            min_dom, max_dom = self.domain[var]
            escala = (max_dom - min_dom) / max_int
            inteiro_esperado = 0.0

            # 𝔼[inteiro] = Σ_j p1(j) · 2^(N-1-j), com j relativo à variável.
            # p1(j) = probabilidade marginal do qubit (var·N + j) ser |1⟩.
            for j in range(N_QUBITS):
                q = var * N_QUBITS + j
                eixos = tuple(e for e in range(self.num_qubits) if e != q)
                marg = amps.sum(axis=eixos)  # vetor de 2 elementos
                p1 = float(marg[1])
                inteiro_esperado += p1 * (2 ** (N_QUBITS - 1 - j))

            media_valores[var] = min_dom + inteiro_esperado * escala

        if dataCollection:
            self.location = media_valores
            # Não armazenar o array de probs completo (2^16 = 512KB/indivíduo):
            # ele não é lido em lugar nenhum e explode o pickling do joblib, que
            # serializa o GA inteiro (incluindo a população) a cada tarefa.
            # Guarda um resumo compacto: os 16 estados mais prováveis.
            top_idx = np.argpartition(probs, -16)[-16:]
            self.simulation_results = {
                format(int(i), f'0{self.num_qubits}b')[::-1]: float(probs[i])
                for i in top_idx
            }

        self.fitness = fitness_function(media_valores)
        return self.fitness


    def _converter_estado_para_reais(self, estado: str) -> List[float]:

        """ Converte uma string binária para uma lista de valores reais dentro do domínio definido. """

        reais = []
        max_int = 2 ** N_QUBITS - 1

        for var in range(self.num_variaveis):

            inicio = var * N_QUBITS

            bits = estado[inicio:inicio + N_QUBITS]
 
            inteiro = int(bits, 2)

            min_dom, max_dom = self.domain[var]

            real = min_dom + (inteiro / max_int) * (max_dom - min_dom)

            reais.append(real)

        return reais


    def construir_circuito_par(self, outro: 'Individuo') -> QuantumCircuit:
        """
        Constrói o circuito conjunto de um par entrelaçado (pairwise entanglement).

        Layout: 2×num_qubits qubits.
          • Qubits [0, num_qubits)          → circuito do `self`
          • Qubits [num_qubits, 2·num_qubits) → circuito do `outro`

        Preparação do estado inicial (o "maximally entangled state" do paper):
          para cada qubit i correspondente, um par de Bell |Φ+⟩ = (|00⟩+|11⟩)/√2
          entre o qubit i do self e o qubit i do outro:
              H(i) ; CX(i, i+num_qubits)
        Depois aplica-se o circuito genético de cada indivíduo nos seus qubits.
        """
        n = self.num_qubits
        if outro.num_qubits != n:
            raise ValueError("indivíduos do par precisam ter o mesmo número de qubits")

        qc = QuantumCircuit(2 * n, 2 * n)

        # 1. Pares de Bell entre qubits correspondentes
        for i in range(n):
            qc.h(i)
            qc.cx(i, i + n)
        qc.barrier()

        # 2. Circuito de cada indivíduo nos seus registradores
        qc_self = self.construir_circuito_qiskit(measure=False)
        qc.compose(qc_self, qubits=list(range(n)), inplace=True)

        qc_outro = outro.construir_circuito_qiskit(measure=False)
        qc.compose(qc_outro, qubits=list(range(n, 2 * n)), inplace=True)

        qc.barrier()
        qc.measure(range(2 * n), range(2 * n))
        return qc


    def calcular_fitness_par(self, outro: 'Individuo',
                             fitness_function: Callable[[List[float]], float],
                             shots: int = 1024, dataCollection: bool = False) -> float:
        """
        Avalia o fitness de um PAR entrelaçado (inter-individual entanglement).

        Simula o circuito conjunto (2× qubits) com os pares de Bell, decodifica
        as marginais de cada indivíduo e calcula f(𝔼[x]) para cada um.

        Returns:
            fitness do `self` (o fitness do `outro` fica em `outro.fitness`).
        """
        qc = self.construir_circuito_par(outro)
        resultados = self.simular_circuito(qc, shots=shots)

        total_shots = sum(resultados.values())
        if total_shots == 0:
            self.fitness = 0.0
            outro.fitness = 0.0
            return self.fitness

        soma_self = [0.0 for _ in range(self.num_variaveis)]
        soma_outro = [0.0 for _ in range(outro.num_variaveis)]

        n = self.num_qubits
        for estado, count in resultados.items():
            rev = estado[::-1]
            vals_self = self._converter_estado_para_reais(rev[:n])
            vals_outro = outro._converter_estado_para_reais(rev[n:])
            for i in range(self.num_variaveis):
                soma_self[i] += vals_self[i] * count
            for i in range(outro.num_variaveis):
                soma_outro[i] += vals_outro[i] * count

        media_self = [s / total_shots for s in soma_self]
        media_outro = [s / total_shots for s in soma_outro]

        if dataCollection:
            self.location = media_self
            outro.location = media_outro
            self.simulation_results = resultados
            outro.simulation_results = resultados

        self.fitness = fitness_function(media_self)
        outro.fitness = fitness_function(media_outro)
        return self.fitness


    def _apply_gate(self, qc: QuantumCircuit, col: int):
        usados = set()
        for q in range(self.num_qubits):
            if q in usados:
                continue
    
            op, c1, c2, t = self.genoma[q, col]
            
            if op == 'K':  # Qubit não usado nesta coluna
                continue
                
            # Portas de 1 qubit
            if op in self.PORTAS1Q:
                if op == 'I':
                    qc.id(t)

                elif op == 'X':
                    qc.x(t)

                elif op == 'Y':
                    qc.y(t)

                elif op == 'Z':
                    qc.z(t)

                elif op == 'H':
                    qc.h(t)

                elif op == 'S':
                    qc.s(t)

                elif op == 'Sd':
                    qc.sdg(t)

                elif op == 'T':
                    qc.t(t)

                elif op == 'Td':
                    qc.tdg(t)

                usados.add(t)
                
            # Portas de 2 qubits
            elif op in self.PORTAS2Q:
                if op == 'CNOT':
                    if(c1 == t):
                        continue
                    qc.cx(c1, t)

                elif op == 'SWAP':
                    if(c1 == t):
                        continue
                    qc.swap(c1, t)

                usados.update([c1, t])
                
            # Portas de 3 qubits
            elif op in self.PORTAS3Q:

                if op == 'TOFF':
                    if(c1 == c2 or c1 == t or c2 == t):
                        # print("controles iguais toff", flush = True)
                        # qc.cx(c1, t)
                        continue
                    else:
                        qc.ccx(c1, c2, t)  # Porta TOFFOLI

                elif op == 'FREAD':
                    if(c1 == c2 or c1 == t or c2 == t):                        
                        #print("controles iguais read", flush = True)
                        continue
                    else:
                        qc.cswap(c1, c2, t)  # Fredkin/Reedkin gate


                usados.update([c1, c2, t])
                
            else:
                print(f'Porta desconhecida: {op} em q{q}')
                qc.id(q)
                usados.add(q)


    def __str__(self):

        def fmt_gene(gene):
            op, c1, c2, t = gene
            if op == 'K':
                return "·    ·    ·    ·"
            c1_s = f"{c1:2d}" if c1 is not None else " ·"
            c2_s = f"{c2:2d}" if c2 is not None else " ·"
            t_s  = f"{t:2d}"  if t  is not None else " ·"
            return f"{op:5s} {c1_s} {c2_s} {t_s}"

        linhas = [f"  • Indivíduo (geração {self.geracao}, fitness={self.fitness}, sigma_es={self.sigma_es:.4f}):\n"]
        for q in range(self.num_qubits):
            linhas.append(f"    ◦ qubit [{q}]:")
            for i in range(self.num_portas):
                linhas.append(f"      gene [{i}]:   {fmt_gene(self.genoma[q, i])}")
            linhas.append("")
        return '\n'.join(linhas)
 

    def _adicionar_colunas(self, num_colunas_para_adicionar):
        """Adiciona novas colunas seguindo a mesma lógica de _gerar_genes_aleatorios"""

        # Cria array para as novas colunas inicializado com 'K'
        novas_colunas = np.array(
            [[('K', None, None, None) for _ in range(num_colunas_para_adicionar)] 
             for q in range(self.num_qubits)],
            dtype=object
        )

        rng = random.Random(self.seed + self.geracao)

        # Para cada nova coluna, aplica a mesma lógica de geração
        for col_idx in range(num_colunas_para_adicionar):
            i = col_idx  # índice da coluna dentro do array de novas colunas

            # Lista de qubits disponíveis nesta coluna (índices)
            disponiveis = [q for q in range(self.num_qubits) if novas_colunas[q, i][0] == 'K']

            while disponiveis:
                q = disponiveis.pop(0)  # Pega o próximo qubit disponível

                # Determinar portas possíveis baseado nos qubits restantes
                possiveis_portas = self.PORTAS1Q.copy()
                qubits_restantes = [x for x in disponiveis if x > q]  # Qubits após o atual

                # Verificar possibilidade de portas de 2 qubits
                if len(qubits_restantes) >= 1 and q <= self.num_qubits - 2:
                    possiveis_portas += self.PORTAS2Q

                # Verificar possibilidade de portas de 3 qubits
                if len(qubits_restantes) >= 2 and q <= self.num_qubits - 3:
                    possiveis_portas += self.PORTAS3Q

                # Escolher porta aleatória dentre as possíveis
                op = rng.choice(possiveis_portas)

                # Aplicar a porta selecionada
                if op in self.PORTAS1Q:
                    novas_colunas[q, i] = (op, None, None, q)

                elif op in self.PORTAS2Q and len(qubits_restantes) >= 1:
                    alvo = rng.choice(qubits_restantes)
                    controle = q
                    if rng.random() < 0.5:  # 50% de chance de inverter os papéis
                        controle, alvo = alvo, q

                    novas_colunas[controle, i] = (op, controle, None, alvo)
                    novas_colunas[alvo, i] = (op, controle, None, alvo)
                    if alvo in disponiveis:
                        disponiveis.remove(alvo)

                elif op in self.PORTAS3Q and len(qubits_restantes) >= 2:
                    qubits_operacao = [q] + rng.sample(qubits_restantes, 2)
                    rng.shuffle(qubits_operacao)
                    controle1, controle2, alvo = qubits_operacao

                    for qubit in [controle1, controle2, alvo]:
                        novas_colunas[qubit, i] = (op, controle1, controle2, alvo)
                        if qubit in disponiveis:
                            disponiveis.remove(qubit)

        return novas_colunas


    def _gene_para_tupla(self, gene):

        if isinstance(gene, tuple) and len(gene) == 4:
            return gene

        if isinstance(gene, list) and len(gene) == 4:
            return tuple(gene)

        if isinstance(gene, np.ndarray) and gene.shape == (4,):
            return tuple(gene.tolist())
        return None


    def _genes_iguais(self, gene, esperado):
        return self._gene_para_tupla(gene) == esperado


    def _indice_qubit_valido(self, qubit):
        return isinstance(qubit, (int, np.integer)) and 0 <= int(qubit) < self.num_qubits


    def _obter_operacao_gene(self, q, i, genes_origem=None):
        genes = genes_origem if genes_origem is not None else self.genoma
        gene = self._gene_para_tupla(genes[q, i])

        if gene is None:
            return None

        op, c1, c2, t = gene
        if op == 'K':
            return None

        if op in self.PORTAS1Q:
            if not self._indice_qubit_valido(t) or int(t) != q:
                return None
            return {'op': op, 'c1': None, 'c2': None, 't': int(t), 'participantes': (q,)}

        if op in self.PORTAS2Q:
            if not (self._indice_qubit_valido(c1) and self._indice_qubit_valido(t)):
                return None
            c1, t = int(c1), int(t)
            if c1 == t or q not in {c1, t}:
                return None
            esperado = (op, c1, None, t)
            if not self._genes_iguais(genes[c1, i], esperado) or not self._genes_iguais(genes[t, i], esperado):
                return None
            return {'op': op, 'c1': c1, 'c2': None, 't': t, 'participantes': (c1, t)}

        if op in self.PORTAS3Q:
            if not (
                self._indice_qubit_valido(c1) and
                self._indice_qubit_valido(c2) and
                self._indice_qubit_valido(t)
            ):
                return None

            c1, c2, t = int(c1), int(c2), int(t)
            participantes = (c1, c2, t)

            if len(set(participantes)) != 3 or q not in participantes:
                return None

            esperado = (op, c1, c2, t)

            if any(not self._genes_iguais(genes[qubit, i], esperado) for qubit in participantes):
                return None

            return {'op': op, 'c1': c1, 'c2': c2, 't': t, 'participantes': participantes}

        return None


    def _portas_mutacao_disponiveis(self, op_antiga):

        if op_antiga in self.PORTAS1Q:

            portas = self.PORTAS1Q.copy()
            if self.num_qubits >= 2:
                portas += self.PORTAS2Q
            if self.num_qubits >= 3:
                portas += self.PORTAS3Q

        elif op_antiga in self.PORTAS2Q:

            portas = self.PORTAS1Q.copy() + self.PORTAS2Q
            if self.num_qubits >= 3:
                portas += self.PORTAS3Q

        elif op_antiga in self.PORTAS3Q:
            portas = self.PORTAS2Q.copy() + self.PORTAS3Q.copy()

        else:
            portas = []

        return [porta for porta in portas if porta != op_antiga]


    def _qubits_1q_disponiveis(self, i, excluidos=None, bloqueados=None):

        excluidos = excluidos or set()
        bloqueados = bloqueados or set()
        candidatos = []

        for qubit in range(self.num_qubits):
            if qubit in excluidos or (qubit, i) in bloqueados:
                continue

            operacao = self._obter_operacao_gene(qubit, i)
            if operacao is not None and operacao['op'] in self.PORTAS1Q:
                candidatos.append(qubit)

        return candidatos
        

    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # abaixo estão as operações de mutação.


    def _aplicar_mutacao_segura(self, q, i, rng, genes_origem=None, bloqueados=None):

        operacao = self._obter_operacao_gene(q, i, genes_origem)
        if operacao is None:
            return set()

        op_antiga = operacao['op']
        portas_disponiveis = self._portas_mutacao_disponiveis(op_antiga)
        if not portas_disponiveis:
            return set()

        nova_porta = rng.choice(portas_disponiveis)
        genes_antes = {qubit: self._gene_para_tupla(self.genoma[qubit, i]) for qubit in range(self.num_qubits)}

        if op_antiga in self.PORTAS1Q:

            if nova_porta in self.PORTAS1Q:
                qubits_afetados = self._mutar_1q_para_1q(q, i, nova_porta)

            elif nova_porta in self.PORTAS2Q:
                qubits_afetados = self._mutar_1q_para_2q(q, i, rng, bloqueados)

            elif nova_porta in self.PORTAS3Q:
                qubits_afetados = self._mutar_1q_para_3q(q, i, nova_porta, rng, bloqueados)

                # Paper Caso 3: se não houver 2 slots 1q na mesma camada,
                # faz fallback para o Caso 1 (substituição 1q → 1q).
                if not qubits_afetados:
                    porta_fallback = rng.choice(self.PORTAS1Q)
                    qubits_afetados = self._mutar_1q_para_1q(q, i, porta_fallback)

            else:
                qubits_afetados = []

        elif op_antiga in self.PORTAS2Q:

            if nova_porta in self.PORTAS1Q:
                qubits_afetados = self._mutar_2q_para_1q(operacao, i, rng)

            elif nova_porta in self.PORTAS2Q:
                qubits_afetados = self._mutar_2q_para_2q(operacao, i, nova_porta)

            elif nova_porta in self.PORTAS3Q:
                qubits_afetados = self._mutar_2q_para_3q(operacao, i, nova_porta, rng, bloqueados)

            else:
                qubits_afetados = []

        elif op_antiga in self.PORTAS3Q:

            if nova_porta in self.PORTAS2Q:
                qubits_afetados = self._mutar_3q_para_2q(operacao, i, rng, nova_porta)

            elif nova_porta in self.PORTAS3Q:
                qubits_afetados = self._mutar_3q_para_3q(operacao, i, nova_porta)

            else:
                qubits_afetados = []

        else:
            qubits_afetados = []

        if not qubits_afetados:
            return set()

        houve_alteracao = any(
            self._gene_para_tupla(self.genoma[qubit, i]) != genes_antes[qubit]
            for qubit in qubits_afetados
        )
        return {(qubit, i) for qubit in qubits_afetados} if houve_alteracao else set()


    def _mutar_1q_para_1q(self, q, i, nova_porta):

        self.genoma[q, i] = (nova_porta, None, None, q)
        return [q]


    def _mutar_1q_para_2q(self, q, i, rng, bloqueados=None):

        outros_qubits = self._qubits_1q_disponiveis(i, excluidos={q}, bloqueados=bloqueados)
        if not outros_qubits:
            return []

        alvo = rng.choice(outros_qubits)
        porta = rng.choice(self.PORTAS2Q)

        self.genoma[q, i]    = (porta, q, None, alvo)
        self.genoma[alvo, i] = (porta, q, None, alvo)

        return [q, alvo]


    def _mutar_1q_para_3q(self, q, i, nova_porta, rng, bloqueados=None):

        # Paper Caso 3: o qubit original vira o CONTROLE (c1) da porta de 3 qubits;
        # os outros dois participantes vêm de slots 1q livres na mesma camada.
        outros_qubits = self._qubits_1q_disponiveis(i, excluidos={q}, bloqueados=bloqueados)
        if len(outros_qubits) < 2:
            return []

        alvo1, alvo2 = rng.sample(outros_qubits, 2)

        # Semântica do gene (op, c1, c2, t):
        #   TOFF: c1,c2 = controles, t = alvo          → q é c1 (controle), alvo1 = c2, alvo2 = t
        #   FREAD: c1 = controle, c2,t = alvos trocados → q é c1 (controle), alvo1/alvo2 = c2,t
        # Em ambos os casos o qubit original q vira o primeiro controle — como no paper.
        gene = (nova_porta, q, alvo1, alvo2)

        for qubit in (q, alvo1, alvo2):
            self.genoma[qubit, i] = gene

        return [q, alvo1, alvo2]


    def _mutar_2q_para_1q(self, operacao, i, rng):

        c1, t = operacao['c1'], operacao['t']

        self.genoma[c1, i]  = (rng.choice(self.PORTAS1Q), None, None, c1)
        self.genoma[t, i]   = (rng.choice(self.PORTAS1Q), None, None, t)

        return [c1, t]


    def _mutar_2q_para_2q(self, operacao, i, nova_porta):

        c1, t = operacao['c1'], operacao['t']

        self.genoma[c1, i]  = (nova_porta, c1, None, t)
        self.genoma[t, i]   = (nova_porta, c1, None, t)

        return [c1, t]


    def _mutar_2q_para_3q(self, operacao, i, nova_porta, rng, bloqueados=None):

        c1, t = operacao['c1'], operacao['t']
        candidatos = self._qubits_1q_disponiveis(i, excluidos={c1, t}, bloqueados=bloqueados)

        if not candidatos:
            return []

        novo_controle = rng.choice(candidatos)
        for qubit in [c1, novo_controle, t]:
            self.genoma[qubit, i] = (nova_porta, c1, novo_controle, t)

        return [c1, novo_controle, t]


    def _mutar_3q_para_2q(self, operacao, i, rng, nova_porta=None):

        c1, c2, t = operacao['c1'], operacao['c2'], operacao['t']

        controle_removido = rng.choice([c1, c2])
        controle_mantido = c2 if controle_removido == c1 else c1

        porta_2q = nova_porta if nova_porta in self.PORTAS2Q else rng.choice(self.PORTAS2Q)

        self.genoma[controle_mantido, i] = (porta_2q, controle_mantido, None, t)
        self.genoma[t, i] = (porta_2q, controle_mantido, None, t)

        self.genoma[controle_removido, i] = (
            rng.choice(self.PORTAS1Q),
            None, None, controle_removido
        )

        return [controle_mantido, t, controle_removido]


    def _mutar_3q_para_3q(self, operacao, i, nova_porta):

        c1, c2, t = operacao['c1'], operacao['c2'], operacao['t']

        for qubit in [c1, c2, t]:
            self.genoma[qubit, i] = (nova_porta, c1, c2, t)

        return [c1, c2, t]
    


    def _adapt_sigma(self, rng: random.Random) -> float:
        """
        Auto-adaptação do sigma_es (Evolution Strategy).
        σ' = σ × exp(τ × N(0,1))
        τ = 1/√(2√n), n = num_variaveis
        """

        n = max(1, self.num_variaveis)
        tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
        
        # N(0,1) via Box-Muller ou random.gauss
        z = rng.gauss(0, 1)
        
        # Auto-adaptação
        novo_sigma = self.sigma_es * math.exp(tau * z)
        
        # Limites razoáveis (evitar degeneração)
        novo_sigma = max(0.01, min(10.0, novo_sigma))
        
        return novo_sigma


    def mutacao(self,
                taxa_mutacao_,
                geracao_atual,
                max_geracoes,
                rng=None):

        if self.num_portas <= 0 or self.num_qubits <= 0:
            return 0


        rng = rng or random.Random(secrets.randbits(32))
        if self.sigma_es_ativo:
            self.sigma_es = self._adapt_sigma(rng)
            taxa_efetiva = taxa_mutacao_ * self.sigma_es
        else:
            taxa_efetiva = taxa_mutacao_

        taxa_mutacao = max(0.0, min(1.0, float(taxa_efetiva)))
        if taxa_mutacao <= 0.0:
            return 0

        max_geracoes = max(1, max_geracoes)
        progresso = max(0.0, min(1.0, geracao_atual / max_geracoes))

        # Janela progressiva de annealing (variante própria, NÃO é do paper).
        # Desligada (annealing_window=False): qualquer gene pode mutar — paper-fiel.
        # Ligada: congela colunas/qubits de alta significância conforme a evolução avança.
        if self.annealing_window:
            J = int(progresso * self.num_portas)
            J = min(J, self.num_portas - 1)

            qubit_inicial_relativo = int(progresso * N_QUBITS)
            qubit_inicial_relativo = min(qubit_inicial_relativo, N_QUBITS - 1)

            qubits_validos = []
            for variavel in range(self.num_variaveis):
                inicio = variavel * N_QUBITS
                fim = inicio + N_QUBITS
                qubits_validos.extend(range(inicio + qubit_inicial_relativo, fim))

            colunas_validas = list(range(J, self.num_portas))
        else:
            qubits_validos = list(range(self.num_qubits))
            colunas_validas = list(range(self.num_portas))

        genes_origem = self.genoma.copy()
        operacoes_tentadas = set()
        posicoes_bloqueadas = set()
        mutacoes_aplicadas = 0

        for q in qubits_validos:
            for i in colunas_validas:
                if rng.random() >= taxa_mutacao:
                    continue

                operacao = self._obter_operacao_gene(q, i, genes_origem)
                if operacao is None:
                    continue

                chave_operacao = (i, operacao['participantes'])
                if chave_operacao in operacoes_tentadas:
                    continue

                posicoes_operacao = {(qubit, i) for qubit in operacao['participantes']}
                if posicoes_operacao & posicoes_bloqueadas:
                    continue

                operacoes_tentadas.add(chave_operacao)
                posicoes_afetadas = self._aplicar_mutacao_segura(
                    q,
                    i,
                    rng,
                    genes_origem=genes_origem,
                    bloqueados=posicoes_bloqueadas
                )

                if posicoes_afetadas:
                    posicoes_bloqueadas.update(posicoes_afetadas)
                    mutacoes_aplicadas += 1

        if mutacoes_aplicadas > 0:
            self.fitness = None

        return mutacoes_aplicadas


    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # abaixo estão os helpers de crossover
    

    def _criar_filho_horizontal(self, outro: 'Individuo', qubit_corte: int, fonte_acima: 'Individuo') -> np.ndarray:
        """
        Cria um genoma de filho por corte horizontal.
        Qubits acima/igual ao corte vêm de fonte_acima; qubits abaixo vêm do outro pai.
        """
        fonte_abaixo = outro if fonte_acima is self else self
        genes = fonte_abaixo.genoma.copy()

        for col in range(self.num_portas):
            for q in range(self.num_qubits):
                if q <= qubit_corte:
                    genes[q, col] = fonte_acima.genoma[q, col]

            self._reparar_coluna_horizontal(genes, col, qubit_corte)

        return genes


    def _reparar_coluna_horizontal(self, genes: np.ndarray, col: int, qubit_corte: int):
        """
        Repara portas de 2 ou 3 qubits que foram cortadas pelo corte horizontal.
        """
        for q in range(self.num_qubits):
            gene = genes[q, col]
            if not isinstance(gene, tuple) or len(gene) != 4:
                continue

            op, c1, c2, t = gene
            if op == 'K' or op not in self.PORTAS2Q + self.PORTAS3Q:
                continue

            if op in self.PORTAS2Q:
                ctrl, target = c1, t
                if (ctrl <= qubit_corte) != (target <= qubit_corte):
                    self._apply_2q_fix(genes, col, op, ctrl, target)

            elif op in self.PORTAS3Q:
                ctrl1, ctrl2, target = c1, c2, t
                lados = (ctrl1 <= qubit_corte, ctrl2 <= qubit_corte, target <= qubit_corte)
                if len(set(lados)) > 1:
                    self._apply_3q_fix(genes, col, op, ctrl1, ctrl2, target, *lados)
    
    
    def _criar_filho_mixed(self, outro: 'Individuo', selected_column1: int,
                           selected_column2: int, cut_points: dict, child1: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria o genoma e o mapa de origens de um filho do mixed_crossover.
        child1=True: qubits <= cut vêm de self; child1=False: qubits <= cut vêm de outro.
        """

        if child1:
            genes = outro.genoma.copy()
            col = selected_column1
            source_above = self.genoma
            source_below = outro.genoma
            source_above_col = selected_column1
            source_below_col = selected_column2
            origin_above = True
            origin_below = False

        else:
            genes = self.genoma.copy()
            col = selected_column2
            source_above = outro.genoma
            source_below = self.genoma
            source_above_col = selected_column2
            source_below_col = selected_column1
            origin_above = False
            origin_below = True

        origins = np.full(self.num_qubits, None)

        for var_idx in range(self.num_variaveis):
            start_qubit = var_idx * N_QUBITS
            end_qubit = start_qubit + N_QUBITS
            cut_qubit = cut_points[var_idx]

            for q in range(start_qubit, end_qubit):
                if q <= cut_qubit:
                    genes[q, col] = source_above[q, source_above_col]
                    origins[q] = origin_above
                else:
                    genes[q, col] = source_below[q, source_below_col]
                    origins[q] = origin_below

        return genes, origins


    def _reparar_coluna_mixed(self, genes: np.ndarray, col: int, origins: np.ndarray):
        """
        Repara portas de 2 ou 3 qubits fragmentadas no mixed_crossover.
        """

        for q in range(self.num_qubits):
            gene = genes[q, col]
            if not isinstance(gene, tuple) or len(gene) != 4:
                continue

            op, c1, c2, t = gene
            if op == 'K' or op not in self.PORTAS2Q + self.PORTAS3Q:
                continue

            if op in self.PORTAS2Q:
                ctrl, target = c1, t
                if origins[ctrl] != origins[target]:
                    self._apply_2q_fix(genes, col, op, ctrl, target)

            elif op in self.PORTAS3Q:

                c1_orig, c2_orig, t_orig = origins[c1], origins[c2], origins[t]
                if not (c1_orig == c2_orig == t_orig):
                    self._apply_3q_fix(genes, col, op, c1, c2, t, c1_orig, c2_orig, t_orig)


    def _apply_2q_fix(self, genes, col, op, ctrl, target):
        """Apply fix for 2-qubit gates."""

        if op == 'CNOT':
            genes[ctrl, col] = ('I', None, None, ctrl)
            genes[target, col] = ('X', None, None, target)

        elif op == 'SWAP':
            genes[ctrl, col] = ('I', None, None, ctrl)
            genes[target, col] = ('I', None, None, target)


    def _apply_3q_fix(self, genes, col, op, ctrl1, ctrl2, target, ctrl1_orig, ctrl2_orig, target_orig):
        """Apply specific fix for 3-qubit gates based on fragmentation pattern."""

        # Skip if any origin is None (not initialized)
        if any(orig is None for orig in [ctrl1_orig, ctrl2_orig, target_orig]):
            for qubit in [ctrl1, ctrl2, target]:
                genes[qubit, col] = ('I', None, None, qubit)
            return


        # Skip if all origins are the same (no fragmentation)
        if ctrl1_orig == ctrl2_orig == target_orig:
            return


        if op == 'TOFF':

            # TOFFOLI gate logic (mantém a mesma)
            if target_orig != ctrl1_orig and target_orig != ctrl2_orig:
                genes[ctrl1, col]   = ('I', None, None, ctrl1)
                genes[ctrl2, col]   = ('I', None, None, ctrl2)
                genes[target, col]  = ('X', None, None, target)

            elif target_orig == ctrl1_orig and target_orig != ctrl2_orig:
                genes[ctrl1, col]   = ('CNOT', ctrl1, None, target)
                genes[target, col]  = ('CNOT', ctrl1, None, target)
                genes[ctrl2, col]   = ('I', None, None, ctrl2)

            elif target_orig == ctrl2_orig and target_orig != ctrl1_orig:
                genes[ctrl2, col]   = ('CNOT', ctrl2, None, target)
                genes[target, col]  = ('CNOT', ctrl2, None, target)
                genes[ctrl1, col]   = ('I', None, None, ctrl1)

            elif ctrl1_orig == ctrl2_orig and ctrl1_orig != target_orig:
                genes[ctrl1, col]   = ('I', None, None, ctrl1)
                genes[ctrl2, col]   = ('I', None, None, ctrl2)
                genes[target, col]  = ('X', None, None, target)

            else:
                for qubit in [ctrl1, ctrl2, target]:
                    genes[qubit, col] = ('I', None, None, qubit)

        elif op == 'FREAD':

            if ctrl2_orig == target_orig:
                # Alvos estão no mesmo filho → viram SWAP

                genes[ctrl2, col]   = ('SWAP', ctrl2, None, target)
                genes[target, col]  = ('SWAP', ctrl2, None, target)

                # Controle vira I
                genes[ctrl1, col]   = ('I', None, None, ctrl1)

            else:

                # Alvos estão em filhos diferentes → todos viram I
                for qubit in [ctrl1, ctrl2, target]:
                    genes[qubit, col] = ('I', None, None, qubit)


    

    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # abaixo estão as operações de crossover propriamente ditas

        
    def pass_sigma_es(self, outro: 'Individuo', filho1: 'Individuo', filho2: 'Individuo', rng: Optional[random.Random] = None):
        """
        Transmite sigma_es dos pais para os filhos via interpolação.
        σ_filho = α × σ_pai1 + (1-α) × σ_pai2
        α ∈ [0,1] aleatório por filho.
        """
        rng = rng or random.Random(secrets.randbits(32))

        alpha1 = rng.random()
        alpha2 = rng.random()

        filho1.sigma_es = alpha1 * self.sigma_es + (1 - alpha1) * outro.sigma_es
        filho2.sigma_es = alpha2 * self.sigma_es + (1 - alpha2) * outro.sigma_es


    def crossover_vertical(self, outro: 'Individuo') -> Tuple['Individuo', 'Individuo']:

        rng = random.Random(self.seed + outro.seed + self.geracao)

        max_ponto = self.num_portas - 1

        if max_ponto < 1:
            
            '''
                Essa condicional está para o erro de max_ponto ou para realmente copiar os
                pais nos filhos (elitismo).
            '''

            filho1 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                               self.door_type, self.domain, self.seed)

            filho2 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                               self.door_type, self.domain, self.seed)

            filho1.genoma = self.genoma.copy()
            filho2.genoma = outro.genoma.copy()

            return filho1, filho2


        corte = rng.randint(1, max_ponto)

        genes1 = np.concatenate((self.genoma[:, :corte], outro.genoma[:, corte:]), axis=1)
        genes2 = np.concatenate((outro.genoma[:, :corte], self.genoma[:, corte:]), axis=1)

        filho1 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)

        filho2 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)

        filho1.genoma = genes1
        filho2.genoma = genes2

        self.pass_sigma_es(outro, filho1, filho2, rng)

        return filho1, filho2


    def crossover_horizontal(self, outro: 'Individuo') -> Tuple['Individuo', 'Individuo']:

        rng = random.Random(self.seed + outro.seed + self.geracao + 1)
        qubit_corte = rng.randint(0, self.num_qubits - 1)

        genes1 = self._criar_filho_horizontal(outro, qubit_corte, fonte_acima=self)
        genes2 = self._criar_filho_horizontal(outro, qubit_corte, fonte_acima=outro)

        filho1 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)

        filho2 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)

        filho1.genoma = genes1
        filho2.genoma = genes2

        self.pass_sigma_es(outro, filho1, filho2, rng)

        return filho1, filho2


    def mixed_crossover(self, outro: 'Individuo') -> Tuple['Individuo', 'Individuo']:

        """
        Crossover misto: corte horizontal por variável em uma coluna específica.
        """

        rng = random.Random(self.seed + outro.seed + self.geracao + 2)

        selected_column1 = rng.randint(1, self.num_portas - 1)
        selected_column2 = rng.randint(1, outro.num_portas - 1)

        cut_points = {
            var_idx: rng.randint(var_idx * N_QUBITS, (var_idx + 1) * N_QUBITS - 1)
            for var_idx in range(self.num_variaveis)
        }

        child1_genes, child1_origin = self._criar_filho_mixed(
            outro, selected_column1, selected_column2, cut_points, child1=True
        )

        child2_genes, child2_origin = self._criar_filho_mixed(
            outro, selected_column1, selected_column2, cut_points, child1=False
        )


        self._reparar_coluna_mixed(child1_genes, selected_column1, child1_origin)
        self._reparar_coluna_mixed(child2_genes, selected_column2, child2_origin)


        filho1 = Individuo(self.geracao + 1, self.max_num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)
        filho2 = Individuo(self.geracao + 1, outro.max_num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)


        filho1.genoma = child1_genes
        filho2.genoma = child2_genes

        filho1.num_portas = self.num_portas
        filho2.num_portas = outro.num_portas

        self.pass_sigma_es(outro, filho1, filho2, rng)

        return filho1, filho2


    def crossover_paper(self, outro: 'Individuo') -> Tuple['Individuo', 'Individuo']:
        """
        Crossover fiel ao paper (arXiv:2511.05254, §2.7).

        Regras do paper:
        1. Uma única camada δ é sorteada da SEGUNDA METADE do circuito
           (camadas iniciais permanecem estáveis).
        2. k ∈ {1..n} sorteado uniformemente (n = qubits por variável).
        3. Troca as portas que atuam nos k qubits MENOS significativos
           da camada δ, junto com portas multi-qubit inteiramente contidas
           na região. Portas que cruzam a fronteira da região NÃO são trocadas.
        """
        rng = random.Random(self.seed + outro.seed + self.geracao + 3)

        d = self.num_portas
        if d < 2 or outro.num_portas < 2:
            # Profundidade pequena demais: copiar pais (elitismo implícito)
            filho1 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                               self.door_type, self.domain, self.seed)
            filho2 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                               self.door_type, self.domain, self.seed)
            filho1.genoma = self.genoma.copy()
            filho2.genoma = outro.genoma.copy()
            return filho1, filho2

        # 1. camada δ na segunda metade do circuito
        primeira_metade = d // 2
        delta = rng.randint(primeira_metade, d - 1)

        # 2. k qubits menos significativos (k ∈ {1..n}, n = qubits por variável)
        k = rng.randint(1, N_QUBITS)
        k = min(k, self.num_qubits)
        limite = self.num_qubits - k  # região = [limite, num_qubits-1]

        def _participantes(genoma, col, q):
            """Retorna o conjunto de qubits da porta em (q, col) ou {q} se 1q/K."""
            gene = self._gene_para_tupla(genoma[q, col])
            if gene is None:
                return None
            op, c1, c2, t = gene
            if op == 'K':
                return None
            if op in self.PORTAS1Q:
                return (t,)
            if op in self.PORTAS2Q:
                return (c1, t)
            if op in self.PORTAS3Q:
                return (c1, c2, t)
            return None

        def _dentro_da_regiao(qubits):
            return all(qb >= limite for qb in qubits)

        def _construtor(base, origem):
            """Filho: base igual a `base`, mas na camada delta a região vem de `origem`."""
            genes = base.genoma.copy()

            # Passo A: apagar as células da região que NÃO pertencem a portas
            # que cruzam a fronteira (estas permanecem intactas no filho).
            for qb in range(limite, self.num_qubits):
                part = _participantes(base.genoma, delta, qb)
                if part is None:
                    continue
                if not _dentro_da_regiao(part):
                    continue  # porta inteiramente na região → será substituída
                # célula simples da região: apaga (marcada p/ troca)
                genes[qb, delta] = ('K', None, None, None)

            # Passo B: escrever as portas de `origem` que estão inteiramente na região
            # e cujas células estão todas livres no filho.
            for qb in range(limite, self.num_qubits):
                part = _participantes(origem.genoma, delta, qb)
                if part is None:
                    continue
                if not _dentro_da_regiao(part):
                    continue  # porta de origem cruza a fronteira → não entra
                gene_origem = self._gene_para_tupla(origem.genoma[qb, delta])
                if gene_origem is None:
                    continue
                livre = True
                for p in part:
                    g = self._gene_para_tupla(genes[p, delta])
                    if g is None or g[0] != 'K':
                        livre = False
                        break
                if livre:
                    for p in part:
                        genes[p, delta] = gene_origem

            return genes

        genes1 = _construtor(self, outro)
        genes2 = _construtor(outro, self)

        filho1 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)
        filho2 = Individuo(self.geracao + 1, self.num_portas, self.num_variaveis,
                           self.door_type, self.domain, self.seed)

        filho1.genoma = genes1
        filho2.genoma = genes2

        self.pass_sigma_es(outro, filho1, filho2, rng)

        return filho1, filho2


    def crossover(self, outro: 'Individuo', method: str = 'vertical') -> Tuple['Individuo', 'Individuo']:

        match method:
            case 'paper':
                return self.crossover_paper(outro)
            case 'horizontal':
                return self.crossover_horizontal(outro)
            case 'mixed':
                return self.mixed_crossover(outro)
            case 'vertical' | _:
                return self.crossover_vertical(outro)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
