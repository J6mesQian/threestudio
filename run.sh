for item in 'apple' 'banana' 'hamburger' 'orange' 'grape' 'pineapple' 'strawberry' 'watermelon' 'kiwi' 'mango'
do
    python launch.py --config configs/dreamfusion-if-triplane.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of ${item}"
done

for item in 'apple' 'banana' 'hamburger' 'orange' 'grape' 'pineapple' 'strawberry' 'watermelon' 'kiwi' 'mango'
do
    python launch.py --config configs/dreamfusion-if-volumegrid.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of ${item}"
done
