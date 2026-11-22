python pdf2json.py --pdf ./doc/DS_stm32f030f4.pdf --output ./STM32
python toc_ranker.py --toc ./STM32/toc.txt --prompt peripheral_prompt.txt --output ./STM32 --concurrent 2