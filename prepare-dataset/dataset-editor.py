import os
import math
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class LoraDatasetEditor:
    def __init__(self, root):
        # # Tentar tela cheia
        # try:
        #     root.state("zoomed")  # Windows
        # except:
        #     root.attributes("-fullscreen", True)  # fallback em outros sistemas

        self.root = root
        self.root.title("LoRA Dataset Editor")

        # Lista de pares na memória (não alteramos os .txt originais)
        self.pares = []
        # Apenas um item selecionado
        self.current_item = None

        # Paginação
        self.page_size = 15
        self.pool_page = 0
        self.final_page = 0

        # Layout principal
        self.frame_top = tk.Frame(root)
        self.frame_esquerda = tk.Frame(root)
        self.frame_direita = tk.Frame(root)
        self.frame_preview = tk.Frame(root, bg='white')

        self.frame_top.pack(side="top", fill="x")
        self.frame_esquerda.pack(side="left", fill="both", expand=True)
        self.frame_direita.pack(side="right", fill="both", expand=True)
        self.frame_preview.pack(side="bottom", fill="both", expand=False)

        # Botoes superiores
        tk.Button(self.frame_top, text="Importar Pares", command=self.importar_pares)\
            .pack(side="left", padx=5, pady=5)
        tk.Button(self.frame_top, text="Adicionar Selecionado à Final", command=self.adicionar_selecionados)\
            .pack(side="left", padx=5, pady=5)
        tk.Button(self.frame_top, text="Remover Selecionado da Final", command=self.remover_selecionados)\
            .pack(side="left", padx=5, pady=5)
        tk.Button(self.frame_top, text="Exportar", command=self.exportar)\
            .pack(side="left", padx=5, pady=5)

        # Spinbox para tamanho de página
        tk.Label(self.frame_top, text="Tamanho da página:").pack(side="left", padx=5)
        self.spin_page_size = tk.Spinbox(self.frame_top, from_=5, to=200, width=5, command=self.atualizar_page_size)
        self.spin_page_size.delete(0, "end")
        self.spin_page_size.insert(0, str(self.page_size))
        self.spin_page_size.pack(side="left", padx=5)

        # Pool (lado esquerdo) + scroll + paginação
        self.canvas_pool = tk.Canvas(self.frame_esquerda)
        self.scrollbar_pool = tk.Scrollbar(self.frame_esquerda, orient="vertical", command=self.canvas_pool.yview)
        self.scrollable_frame_pool = tk.Frame(self.canvas_pool)
        self.scrollable_frame_pool.bind(
            "<Configure>",
            lambda e: self.canvas_pool.configure(scrollregion=self.canvas_pool.bbox("all"))
        )
        self.canvas_pool.create_window((0, 0), window=self.scrollable_frame_pool, anchor="nw")
        self.canvas_pool.configure(yscrollcommand=self.scrollbar_pool.set)
        self.scrollable_frame_pool.bind("<MouseWheel>", self.on_pool_mousewheel)

        self.canvas_pool.pack(side="left", fill="both", expand=True)
        self.scrollbar_pool.pack(side="right", fill="y")

        self.pool_pagination_frame = tk.Frame(self.frame_esquerda)
        self.pool_pagination_frame.pack(side="bottom", fill="x")

        # Seleção final (lado direito) + scroll + paginação
        self.canvas_final = tk.Canvas(self.frame_direita)
        self.scrollbar_final = tk.Scrollbar(self.frame_direita, orient="vertical", command=self.canvas_final.yview)
        self.scrollable_frame_final = tk.Frame(self.canvas_final)
        self.scrollable_frame_final.bind(
            "<Configure>",
            lambda e: self.canvas_final.configure(scrollregion=self.canvas_final.bbox("all"))
        )
        self.canvas_final.create_window((0, 0), window=self.scrollable_frame_final, anchor="nw")
        self.canvas_final.configure(yscrollcommand=self.scrollbar_final.set)
        self.scrollable_frame_final.bind("<MouseWheel>", self.on_final_mousewheel)

        self.canvas_final.pack(side="left", fill="both", expand=True)
        self.scrollbar_final.pack(side="right", fill="y")

        self.final_pagination_frame = tk.Frame(self.frame_direita)
        self.final_pagination_frame.pack(side="bottom", fill="x")

        # Preview maior
        self.label_preview_img = tk.Label(self.frame_preview, bg='white')
        self.label_preview_img.pack(side="left", padx=10, pady=10)
        self.text_preview_desc = tk.Text(self.frame_preview, height=8, width=40)
        self.text_preview_desc.pack(side="left", padx=10, pady=10)
        tk.Button(self.frame_preview, text="Salvar Descrição do Selecionado", command=self.salvar_descricao)\
            .pack(side="left", padx=5, pady=5)

    # --------------------------------------------------------------------------------
    #                            PAGINAÇÃO E SCROLL
    # --------------------------------------------------------------------------------
    def atualizar_page_size(self):
        try:
            tamanho = int(self.spin_page_size.get())
            if tamanho <= 0:
                raise ValueError
            self.page_size = tamanho
        except:
            self.page_size = 15
            self.spin_page_size.delete(0, "end")
            self.spin_page_size.insert(0, "15")
        self.pool_page = 0
        self.final_page = 0
        self.render_frames()

    def on_pool_mousewheel(self, event):
        pool_items = [p for p in self.pares if not p['selecionado']]
        max_page = self.calc_max_page(len(pool_items))
        delta = self.get_mouse_delta(event)
        if delta > 0:  # subir página
            self.pool_page = max(0, self.pool_page - 1)
        else:          # descer página
            self.pool_page = min(max_page, self.pool_page + 1)
        self.render_frames()

    def on_final_mousewheel(self, event):
        final_items = [p for p in self.pares if p['selecionado']]
        max_page = self.calc_max_page(len(final_items))
        delta = self.get_mouse_delta(event)
        if delta > 0:
            self.final_page = max(0, self.final_page - 1)
        else:
            self.final_page = min(max_page, self.final_page + 1)
        self.render_frames()

    def calc_max_page(self, total_items):
        if self.page_size <= 0:
            return 0
        return max(0, (total_items - 1) // self.page_size)

    def get_mouse_delta(self, event):
        delta = event.delta
        if delta == 0 and hasattr(event, 'num'):
            # Linux fallback
            if event.num == 4:
                delta = 120
            elif event.num == 5:
                delta = -120
        return delta

    def ir_para_pagina_pool(self, page):
        self.pool_page = page
        self.render_frames()

    def ir_para_pagina_final(self, page):
        self.final_page = page
        self.render_frames()

    # --------------------------------------------------------------------------------
    #                         IMPORTAR E RENDERIZAR
    # --------------------------------------------------------------------------------
    def importar_pares(self):
        file_paths = filedialog.askopenfilenames(
            title="Selecione imagens",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg")]
        )
        if not file_paths:
            return

        for img_path in file_paths:
            base = os.path.splitext(img_path)[0]
            txt_path = base + ".txt"
            if os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    descricao = f.read()
            else:
                descricao = ""
            self.pares.append({
                'imagem': img_path,
                'descricao': descricao,
                'selecionado': False
            })
        self.pool_page = 0
        self.final_page = 0
        self.render_frames()

    def render_frames(self):
        # Limpar frames do pool
        for w in self.scrollable_frame_pool.winfo_children():
            w.destroy()
        # Limpar frames do final
        for w in self.scrollable_frame_final.winfo_children():
            w.destroy()
        # Limpar frames de paginação
        for w in self.pool_pagination_frame.winfo_children():
            w.destroy()
        for w in self.final_pagination_frame.winfo_children():
            w.destroy()

        # Obter itens pool/final e paginar
        pool_items = sorted([p for p in self.pares if not p['selecionado']],
                            key=lambda p: os.path.basename(p['imagem']).lower())
        final_items = sorted([p for p in self.pares if p['selecionado']],
                             key=lambda p: os.path.basename(p['imagem']).lower())

        # Pool
        pool_total = len(pool_items)
        max_page_pool = self.calc_max_page(pool_total)
        start_idx = self.pool_page * self.page_size
        end_idx = start_idx + self.page_size
        page_pool_items = pool_items[start_idx:end_idx]

        # Final
        final_total = len(final_items)
        max_page_final = self.calc_max_page(final_total)
        start_idx_f = self.final_page * self.page_size
        end_idx_f = start_idx_f + self.page_size
        page_final_items = final_items[start_idx_f:end_idx_f]

        # Cabeçalhos
        tk.Label(self.scrollable_frame_pool, text="Pool de Pares", font=("Arial", 12, "bold"))\
            .pack(anchor="w", pady=5)
        tk.Label(self.scrollable_frame_final, text="Seleção Final", font=("Arial", 12, "bold"))\
            .pack(anchor="w", pady=5)

        # Renderizar Pool (página atual)
        for par in page_pool_items:
            self.criar_thumbnail_frame(self.scrollable_frame_pool, par)

        # Renderizar Final (página atual)
        for par in page_final_items:
            self.criar_thumbnail_frame(self.scrollable_frame_final, par)

        # Criar paginador do pool
        self.criar_paginador(self.pool_pagination_frame,
                             self.pool_page,
                             max_page_pool,
                             self.ir_para_pagina_pool,
                             prefixo="Pool:",
                             container='pool')

        # Criar paginador do final
        self.criar_paginador(self.final_pagination_frame,
                             self.final_page,
                             max_page_final,
                             self.ir_para_pagina_final,
                             prefixo="Final:",
                             container='final')

    def criar_thumbnail_frame(self, parent, par):
        """Cria o frame contendo thumbnail, nome do arquivo e 1 linha da descrição."""
        frame_item = tk.Frame(parent, bd=2, relief="groove")
        frame_item.pack(fill="x", pady=2, padx=2)

        thumb = self.get_thumbnail(par['imagem'], size=64)
        lbl_img = tk.Label(frame_item, image=thumb)
        lbl_img.image = thumb  # evitar GC
        lbl_img.pack(side="left", padx=5, pady=5)

        # Lado texto
        txt_frame = tk.Frame(frame_item)
        txt_frame.pack(side="left", fill="both", expand=True)

        # Nome do arquivo
        nome_arquivo = os.path.basename(par['imagem'])
        lbl_nome = tk.Label(txt_frame, text=nome_arquivo)
        lbl_nome.pack(anchor="w")

        # Descrição em 1 linha (truncada)
        desc_line = par['descricao'].replace('\n', ' ')
        if len(desc_line) > 60:
            desc_line = desc_line[:60] + "..."
        lbl_desc = tk.Label(txt_frame, text=desc_line, fg="gray")
        lbl_desc.pack(anchor="w")

        # Clique => define current_item
        frame_item.bind("<Button-1>", lambda e, p=par: self.on_select_item(p))
        lbl_img.bind("<Button-1>", lambda e, p=par: self.on_select_item(p))
        lbl_nome.bind("<Button-1>", lambda e, p=par: self.on_select_item(p))
        lbl_desc.bind("<Button-1>", lambda e, p=par: self.on_select_item(p))

    def adicionar_todos_pool_page(self):
        """Adiciona todos os pares da página atual (pool) na Seleção Final."""
        pool_items = sorted([p for p in self.pares if not p['selecionado']],
                            key=lambda p: os.path.basename(p['imagem']).lower())
        start_idx = self.pool_page * self.page_size
        end_idx = start_idx + self.page_size
        page_pool_items = pool_items[start_idx:end_idx]
        for p in page_pool_items:
            p['selecionado'] = True
        self.pool_page = 0
        self.final_page = 0
        self.render_frames()

    def remover_todos_final_page(self):
        """Remove todos os pares da página atual (final) - voltando para a pool."""
        final_items = sorted([p for p in self.pares if p['selecionado']],
                            key=lambda p: os.path.basename(p['imagem']).lower())
        start_idx = self.final_page * self.page_size
        end_idx = start_idx + self.page_size
        page_final_items = final_items[start_idx:end_idx]
        for p in page_final_items:
            p['selecionado'] = False
        self.pool_page = 0
        self.final_page = 0
        self.render_frames()

    def criar_paginador(self, parent, pagina_atual, pagina_maxima, callback, prefixo="", container=None):
        """Cria os botões de navegação (<< < > >>) e exibe página atual."""
        tk.Label(parent, text=f"{prefixo} Página {pagina_atual+1} de {pagina_maxima+1}").pack(side="left", padx=5)

        btn_inicio = tk.Button(parent, text="<<", command=lambda: callback(0))
        btn_inicio.pack(side="left", padx=2)

        btn_voltar = tk.Button(parent, text="<", command=lambda: callback(max(0, pagina_atual - 1)))
        btn_voltar.pack(side="left", padx=2)

        btn_avancar = tk.Button(parent, text=">", command=lambda: callback(min(pagina_atual + 1, pagina_maxima)))
        btn_avancar.pack(side="left", padx=2)

        btn_fim = tk.Button(parent, text=">>", command=lambda: callback(pagina_maxima))
        btn_fim.pack(side="left", padx=2)

        # Se for container='pool', criamos o botão "Adicionar todos"
        if container == 'pool':
            btn_all_pool = tk.Button(parent, text="Adicionar todos da página",
                                    command=self.adicionar_todos_pool_page)
            btn_all_pool.pack(side="right", padx=5)

        # Se for container='final', criamos o botão "Remover todos"
        elif container == 'final':
            btn_all_final = tk.Button(parent, text="Remover todos da página",
                                    command=self.remover_todos_final_page)
            btn_all_final.pack(side="right", padx=5)

    # --------------------------------------------------------------------------------
    #                              SELEÇÃO E PREVIEW
    # --------------------------------------------------------------------------------
    def on_select_item(self, par):
        self.current_item = par
        self.update_preview()

    def update_preview(self):
        if self.current_item:
            try:
                big_img = Image.open(self.current_item['imagem'])
                big_img.thumbnail((400, 400))
                self.tk_big_img = ImageTk.PhotoImage(big_img)
                self.label_preview_img.config(image=self.tk_big_img, text='')
            except:
                self.label_preview_img.config(image='', text='Erro ao carregar imagem')

            self.text_preview_desc.delete("1.0", tk.END)
            self.text_preview_desc.insert(tk.END, self.current_item['descricao'])
        else:
            self.label_preview_img.config(image='', text='(Nenhum item selecionado)')
            self.text_preview_desc.delete("1.0", tk.END)

    # --------------------------------------------------------------------------------
    #                  AÇÕES: ADICIONAR, REMOVER, SALVAR DESCRIÇÃO
    # --------------------------------------------------------------------------------
    def adicionar_selecionados(self):
        """Move o item atual para a Seleção Final."""
        if self.current_item and not self.current_item['selecionado']:
            self.current_item['selecionado'] = True
            self.pool_page = 0
            self.final_page = 0
            self.render_frames()

    def remover_selecionados(self):
        """Remove o item atual da Seleção Final."""
        if self.current_item and self.current_item['selecionado']:
            self.current_item['selecionado'] = False
            self.pool_page = 0
            self.final_page = 0
            self.render_frames()

    def salvar_descricao(self):
        """Apenas atualiza a descrição em memória, sem tocar o .txt original."""
        if not self.current_item:
            return
        novo_texto = self.text_preview_desc.get("1.0", tk.END).strip()
        self.current_item['descricao'] = novo_texto
        # Atualiza a UI para refletir a descrição truncada
        self.render_frames()
        # messagebox.showinfo("Salvar Descrição", "A descrição foi atualizada na memória.")

    # --------------------------------------------------------------------------------
    #                                 EXPORTAÇÃO
    # --------------------------------------------------------------------------------
    def exportar(self):
        """Exporta usando a descrição em memória (sem alterar o .txt original)."""
        output_dir = filedialog.askdirectory()
        if not output_dir:
            return

        epochs = self.simple_input("Número de epochs:")
        trigger = self.simple_input("Trigger word:")
        categorias = self.simple_input("Categorias (separadas por espaço):")

        if not epochs or not trigger:
            messagebox.showwarning("Atenção", "É necessário informar epochs e trigger.")
            return

        # Montar estrutura
        img_dir = os.path.join(output_dir, "img")
        log_dir = os.path.join(output_dir, "log")
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        subpasta_nome = f"{epochs}_{trigger} {categorias}"
        subpasta_path = os.path.join(img_dir, subpasta_nome)
        os.makedirs(subpasta_path, exist_ok=True)

        # Copiar itens selecionados, criando novos .txt com a descrição atual
        selecionados = [p for p in self.pares if p['selecionado']]
        for par in selecionados:
            img_basename = os.path.basename(par['imagem'])
            txt_basename = os.path.splitext(img_basename)[0] + ".txt"

            destino_img = os.path.join(subpasta_path, img_basename)
            destino_txt = os.path.join(subpasta_path, txt_basename)

            shutil.copy2(par['imagem'], destino_img)
            with open(destino_txt, "w", encoding="utf-8") as f:
                f.write(par['descricao'])

        messagebox.showinfo("Exportar", "Exportação concluída com sucesso.")

    # --------------------------------------------------------------------------------
    #                          FUNÇÕES DE UTILIDADE
    # --------------------------------------------------------------------------------
    def simple_input(self, prompt_text):
        def confirm():
            nonlocal value
            value = entry.get()
            win.destroy()

        value = ""
        win = tk.Toplevel(self.root)
        win.title(prompt_text)
        tk.Label(win, text=prompt_text).pack(side="top", padx=10, pady=5)
        entry = tk.Entry(win)
        entry.pack(side="top", padx=10, pady=5)
        btn = tk.Button(win, text="OK", command=confirm)
        btn.pack(side="top", pady=5)

        entry.focus()
        win.grab_set()
        win.wait_window()
        return value

    def get_thumbnail(self, path, size=64):
        """Retorna um objeto PhotoImage (thumbnail)."""
        try:
            im = Image.open(path)
            im.thumbnail((size, size))
            return ImageTk.PhotoImage(im)
        except:
            # Se falhar, devolve um quadrado cinza
            temp = Image.new("RGB", (size, size), color=(128,128,128))
            return ImageTk.PhotoImage(temp)


if __name__ == "__main__":
    root = tk.Tk()
    app = LoraDatasetEditor(root)
    root.mainloop()
