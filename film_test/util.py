import matplotlib.pyplot as plt
import torch

from film_test.cifar import denormalize, QUESTIONS, classes


def make_debug_qa_diagram(img, question_idx, target, pred, img_class, epoch_no, comet):
    img = denormalize(img[0]).permute(1, 2, 0).cpu().numpy()
    print(type(img))
    print(img.shape)
    print (question_idx)
    question_idx = torch.argmax(question_idx).item()
    print(question_idx)
    question = QUESTIONS[question_idx]
    right_answer = "Yes" if target.item() == 1 else "No"
    predicted_answer = "Yes" if pred.item() == 1 else "No"
    img_class = classes[img_class.item()]
    plt.imshow(img)
    plt.title(f"Epoch {epoch_no}, Question: {question}\n"
              f"Correct Answer: {right_answer}, Predicted: {predicted_answer}\n"
              f"Image depicts: {img_class}")
    plt.axis('off')
    comet.log_figure(figure=plt)

