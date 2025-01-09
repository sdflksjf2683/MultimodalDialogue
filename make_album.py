import json, os, re
from openai import OpenAI

# (원본) Image_utils 에서 필요한 함수 임포트
from Image_utils import (
    generate_cosmicman_image,  # 새로 추가한 CosmicMan-SDXL 함수
    generate_landscape_image,
    generate_uniportrait_image,
    generate_uniportrait_image_with_2_faces,
    generate_uniportrait_image_with_3_faces,
    generate_uniportrait_image_with_4_faces,
    generate_uniportrait_image_with_5_faces
)

# CUDA 환경 설정 (필요시)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


##########################################################################
# 1. 다중 얼굴 이미지를 한 번에 생성하는 통합 함수 (uniportrait)
##########################################################################
def generate_uniportrait_image_with_multiple_faces(
    prompt,
    negative_prompt,
    face_image_paths,
    output_path,
    faceid_scale=0.7,
    face_structure_scale=0.6,
    num_samples=1,
    seed=42,
    resolution=(512, 512),
    steps=80
):
    """
    face_image_paths: 실제로 매칭된 얼굴 이미지 경로들의 리스트 (최대 5개)
    """
    num_faces = len(face_image_paths)

    if num_faces == 0:
        print("[WARN] 매칭된 얼굴이 없습니다. 스킵합니다.")
        return

    if num_faces == 1:
        generate_uniportrait_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image_path=face_image_paths[0],
            output_path=output_path,
            faceid_scale=faceid_scale,
            face_structure_scale=face_structure_scale,
            num_samples=num_samples,
            seed=seed,
            resolution=resolution,
            steps=steps
        )

    elif num_faces == 2:
        generate_uniportrait_image_with_2_faces(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image_path_1=face_image_paths[0],
            face_image_path_2=face_image_paths[1],
            output_path=output_path,
            faceid_scale=faceid_scale,
            face_structure_scale=face_structure_scale,
            num_samples=num_samples,
            seed=seed,
            resolution=resolution,
            steps=steps
        )

    elif num_faces == 3:
        generate_uniportrait_image_with_3_faces(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image_path_1=face_image_paths[0],
            face_image_path_2=face_image_paths[1],
            face_image_path_3=face_image_paths[2],
            output_path=output_path,
            faceid_scale=faceid_scale,
            face_structure_scale=face_structure_scale,
            num_samples=num_samples,
            seed=seed,
            resolution=resolution,
            steps=steps
        )

    elif num_faces == 4:
        generate_uniportrait_image_with_4_faces(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image_path_1=face_image_paths[0],
            face_image_path_2=face_image_paths[1],
            face_image_path_3=face_image_paths[2],
            face_image_path_4=face_image_paths[3],
            output_path=output_path,
            faceid_scale=faceid_scale,
            face_structure_scale=face_structure_scale,
            num_samples=num_samples,
            seed=seed,
            resolution=resolution,
            steps=steps
        )

    elif num_faces == 5:
        generate_uniportrait_image_with_5_faces(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image_path_1=face_image_paths[0],
            face_image_path_2=face_image_paths[1],
            face_image_path_3=face_image_paths[2],
            face_image_path_4=face_image_paths[3],
            face_image_path_5=face_image_paths[4],
            output_path=output_path,
            faceid_scale=faceid_scale,
            face_structure_scale=face_structure_scale,
            num_samples=num_samples,
            seed=seed,
            resolution=resolution,
            steps=steps
        )

    else:
        print(f"[WARN] 현재 5명을 초과하는 얼굴은 지원하지 않습니다. ({num_faces} faces)")
        return


##########################################################################
# 2. 기존 보조 함수들
##########################################################################

def remove_prefix(input_string):
    # ':'를 기준으로 분리하고 뒤쪽만 반환
    return input_string.split(": ", 1)[-1]

def get_response(prompt1, prompt2, model_name, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"{prompt1}",
            },
            {
                "role": "user",
                "content": f"{prompt2}",
            },
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    return completion.choices[0].message.content

def make_profile(name, dialogue, image_description, speaker, persona, model_name, api_key):
    name = name.replace("*", "").strip()
    prompt1 = f'''You will be provided with an image description, persona and a conversation between two individuals. Using this information, create a detailed profile for {name}. 
    The persona belongs to {speaker}. When creating a profile, consider incorporating the persona's information if applicable
    You need to analyze the conversation and identify who {name} is, then present the results in the format below:
    Each profile must include name, age, nationality (or species, if applicable), gender, physical description, occupation (if applicable), and personality traits. Use descriptive adjectives to elaborate on their physical appearance, following this format: Description + Variety of Adjectives + Style. When describing their appearance, use only adjectives. Avoid using phrases like "Not specified" or "Unknown." If specific details are missing, infer logical and definitive information based on the context of the image and conversation, ensuring all profiles are complete and coherent. Additionally, describe the relationship between the individuals as specifically and expansively as possible, using all provided information. 
    Write a detailed description specifying the exact species of the animal. If the name is inaccurate, return the result as None.
    The output format should be as follows:
    <{name}'s profile>
    -{name}'s name :
    -{name}'s age :
    -{name}'s nationality or species : 
    -{name}'s gender :
    -{name}'s physical description :
    -{name}'s occupation :
    -{name}'s personality traits :
    -{name}'s role in the relationship

    [Example]
    <Nate's profile> 
    -Nate's name: Nathaniel Gray,
    -Nate's age: 28,
    -Nate's nationality or species: British-American,
    -Nate's gender:  Male,
    -Nate's physical_description: Athletic, Tall, Vibrant, Frequently seen in casual athletic wear, with an enthusiastic demeanor and a quick smile that speaks of his love for exploration.
    -Nate's occupation: Marketing Assistant at a startup in San Francisco
    -Nate's personality: Adventurous, Creative, Quirky, Driven, Warm-hearted, Energetic.
    -Nate's role in the relationship: Nate often brings creativity and excitement into the marriage, encouraging Emily to embrace the unconventional and pushing her to explore new experiences beyond her career.'''
    prompt2 = f'''
image_desrciption : {image_description}
dialogue : {dialogue}
persona : {persona}
Now, based on given information, craft the profile of {name}:
'''
    profile = get_response(prompt1, prompt2, model_name, api_key)
    return profile

def parse_speaker_information(result):
    def extract_information(start_index):
        keys = [
            'name',
            'age',
            'nationality or species',
            'gender',
            'physical_description',
            'occupation',
            'personality',
            'role'
        ]
        return {
            key: result[start_index + i].split(":")[1].strip()
            for i, key in enumerate(keys)
        }

    information = extract_information(0)
    return information

def process_relationship_text(relationship_text):
    lines = relationship_text.split('\n')
    result = []
    current_item = ""

    for line in lines:
        if line.startswith('-') or line.startswith('<'):
            if current_item.strip():
                result.append(current_item.strip())
            current_item = line
        else:
            current_item += f" {line}"

    if current_item.strip():
        result.append(current_item.strip())

    return [item for item in result if '-' in item]

def split_into_entries(input_string):
    entries = re.findall(r'<[^>]+>', input_string)
    return entries


##########################################################################
# 3. search_acquaintance 함수
##########################################################################
def search_acquaintance(data, model_name, api_key):
    prompt1 = '''You will be provided with a double list and conversation between two individuals. Your task is to filter it by removing lists that do not have an object name, and simply perform the filtering
    Filter the data based on the following instruction:
    - After filtering, if there are multiple descriptions for the same object, consolidate them into a summarized format and organize as follows
    - Exclude cases where the description refers to a person but uses a generic noun (e.g., "people," "grandparent") instead of a proper name.
    - Whenever a person's name appears, enclose it in <> and organize it using a detailed description.
    - Carefully filter through the nested lists to identify and distinguish between names of people and animals.
    - if the result is empty, return <>.
    The output format should be as follows:
    - <Name : Description> 

    [Example]
    dialogue : Alejandro: Hey Ethan, how's everything in the land of majestic rock formations and serene solitude?\n\nEthan: Hey Alejandro. It's been a peaceful morning over here, just got back from a sunrise hike at Bryce Canyon. The views were unbelievable! How about you, catching inspiration in the bustling streets of Toronto?\n\nAlejandro: Absolutely! Just finished a new piece inspired by the Distillery District. The lights and the atmosphere give me so much to work with. Here, let me show you—\n\nAlejandro:<turn><landscape><2022-02-04 6:00 AM><A vibrant photograph capturing the dazzling lights and art installations of Toronto's Distillery District, with colorful reflections in the rain-soaked cobblestones.>\n\nEthan: That's stunning, Alejandro! It almost feels like I'm walking through there. The reflections give it such a lively vibe. Reminds me of the street murals I saw last month in Salt Lake City.\n\nAlejandro: Thanks, man. Those murals must have been something! I can imagine them having that Miami touch you mentioned before. Speaking of which, still crafting those unique frames?\n\nEthan: Oh, you bet. I recently finished a new one using some reclaimed barn wood. It's got that rugged charm. I'll send you a pic.\n\nEthan:<turn><people><2022-02-03 8:45 AM><A rustic picture frame made from reclaimed barn wood, showcasing Ethan’s craftsmanship with intricate carvings and a softly varnished finish.>\n\nAlejandro: Wow, Ethan, the craftsmanship is impeccable. It’s got a real old-world charm, like it could hold a family portrait from the 1900s. Your skill with wood always amazes me.\n\nEthan: Thanks, Alejandro. I guess it's my way of preserving a piece of history. Speaking of history, any new travel plans on your horizon?\n\nAlejandro: Actually, Montreal is calling my name. There's this festival next month that blends art and music. It's like a creative paradise. What about you, any trips on the radar?\n\nEthan: That sounds fantastic! I've been eyeing Grand Staircase-Escalante. Its vastness and the natural beauty there have been on my list for ages. Maybe I’ll find some inspiration to bring home.\n\nAlejandro: That sounds just as magical! And you never know, maybe you’ll find a piece of rock to craft into something epic. You’ve always had a knack for transforming nature into art.\n\nEthan: Ha, we'll see! It might be time for something new. Oh, speaking of new, check out this mural I found. It made me think of your Miami days.\n\nEthan:<turn><landscape><2022-02-02 10:15 AM><A vibrant street mural in Salt Lake City, featuring dynamic colors and intricate patterns depicting a fusion of nature and city life, reminiscent of Art Deco style.>\n\nAlejandro: The colors are so vibrant! It does have that Miami flair. I love how it melds the city and nature seamlessly. You’ve got an eye for finding these gems, Ethan.\n\nEthan: Thanks, Alejandro. We definitely share an eye for making the ordinary extraordinary. Can’t wait to hear about your Montreal experience. Maybe you’ll find a new muse there.\n\nAlejandro: For sure, and maybe I’ll send you a piece from the trip. Our adventures always seem to create incredible stories. Until then, keep crafting and exploring!\n\nEthan: You too, Alejandro. Here’s to more stories and epic journeys!",
    Double List : [['people', 'A rustic picture frame made from reclaimed barn wood, showcasing Ethan’s craftsmanship with intricate carvings and a softly varnished finish.']]
    output : <>

    dialogue : 
    Double List :  [['Spike', 'A cute photo of me with Spike, my bearded dragon, perched on my shoulder, both enjoying the sunlight in the garden.'], ['Max', 'A playful picture of me and Max, an excited golden retriever with a squeaky toy, in a sunlit backy
ard.'], ['Spike', 'A cute snapshot of Spike, my bearded dragon, perched on my shoulder as I smile, surrounded by the vibrant greenery of my backyard garden.'], ['Max', "A warm photo of Max, the golden retriever, playfully nuzzling my hand in my backyard,
 with Sedona's stunning red rock formations in the background."], ['Spike', 'A photo of Hayden lounging on a sunny sofa with Spike the bearded dragon perched comfortably on his shoulder.'], ['Max', "A delightful photo of Max, my neighbor's golden retriev
er, playfully bounding in the Sedona red rocks, joyfully mid-leap."], ['Max', 'A playful photo of Marissa and Max, the golden retriever, in her backyard, with Max trying to nibble on a garden hose.'], ['Oliver the cat', "A quirky photo of Thomas with Oliver, his feisty tabby cat, playfully batting at Thomas's finger, with a playful expression and a glint of mischief in its eyes."], ['Oliver', 'A playful snapshot of Oliver the tabby ca
t and the two rescue dogs, Rocky and Bella, surrounded by lush greenery during a hike.']]
    output : 
    <Spike: Spike, the bearded dragon, is frequently seen perched on shoulders, enjoying sunny and vibrant environments, whether in a garden or indoors.>
    <Max: Max, the golden retriever, is a playful and energetic companion often found enjoying outdoor adventures, backyard fun, or the stunning scenery of Sedona.>
    <Oliver: Oliver, the tabby cat, is a spirited and playful companion, often engaging in mischievous antics and enjoying lush outdoor settings with other animals.>'''
    for dialogues in data:
        acquaintance = []
        speaker1 = dialogues['speaker1']['name'].strip()
        speaker1_persona = dialogues['session_1'][f"{speaker1}'s persona"]
        speaker2 = dialogues['speaker2']['name'].strip()
        speaker2_persona = dialogues['session_1'][f"{speaker2}'s persona"]

        folder_name = f"{speaker1}-{speaker2}"
        main_folder_path = os.path.join('./generated_images', folder_name)
        face_folder_path = os.path.join(main_folder_path, 'face')

        people_album = []
        for num in range(1, 6):
            session_key = f'session_{num}'
            dialogue = dialogues[session_key]['dialogue']
            dialogue_all = dialogues[session_key]['all_dia']

            for utter in dialogue:
                if 'photo_description' in list(utter.keys()) and 'landscape' != utter['people']:
                    people = [p.strip() for p in re.sub(r'\band\b', ',', utter['people']).strip().split(',')]
                    for person in people:
                        if person != 'me' and person not in speaker1 and person not in speaker2:
                            people_album.append([f'{person}', utter['photo_description'], dialogue_all])

        prompt2 = f'''Filter the data carefully and logically, ensuring a thorough and thoughtful approach.
        Double List : {[[i[0], i[1]] for i in people_album]}'''
        filter_result = get_response(prompt1, prompt2, model_name, api_key)
        face_list = split_into_entries(filter_result)
        for i, person in enumerate(face_list):
            name = person.split(":")[0].strip('<>').strip()
            description = person.split(":")[1].strip('<>').strip()
            for album in people_album:
                if album[0] == name:
                    face_list[i] = f"{name} ### {description} ### {album[2]}"

        for person in face_list:
            try:
                name = person.split("###")[0].strip('<>').strip()
                description = person.split("###")[1].strip('<>').strip()
                dialogue_all = person.split("###")[2].strip('<>').strip()
            except:
                continue

            person_file_name = f"{name}.png"
            person_image_path = os.path.join(face_folder_path, person_file_name)

            if utter['speaker'] in speaker1:
                persona = speaker1_persona
                speaker = speaker1
            else:
                persona = speaker2_persona
                speaker = speaker2

            profile = make_profile(
                name=name,
                dialogue=dialogue_all,
                image_description=description,
                speaker=speaker,
                persona=persona,
                model_name=model_name,
                api_key=api_key
            ).strip()

            profile = process_relationship_text(profile)
            information = parse_speaker_information(profile)
            acquaintance.append(information)

            file_path = os.path.join(main_folder_path, 'acquaintance.json')
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(acquaintance, json_file, ensure_ascii=False, indent=4)


def make_face_acquaintance(data):
    for dialogues in data:
        speaker1 = dialogues['speaker1']['name'].strip()
        speaker2 = dialogues['speaker2']['name'].strip()

        folder_name = f"{speaker1}-{speaker2}"
        main_folder_path = os.path.join('./generated_images', folder_name)
        face_folder_path = os.path.join(main_folder_path, 'face')

        acquaintance_file_path = os.path.join(main_folder_path, 'acquaintance.json')
        if not os.path.exists(acquaintance_file_path):
            continue

        with open(acquaintance_file_path, 'r', encoding='utf-8') as f:
            acquaintances = json.load(f)

        for acquaintance in acquaintances:
            name = acquaintance['name']
            physical_description = acquaintance['physical_description']

            person_image_path = os.path.join(face_folder_path, f"{name}.png")
            prompt = "A realistic photograph of a human, upper body and face"
            negative_prompt = (
                "hands, arms, legs, feet, torso, cropped head, cropped face, out of frame, blurry, distorted, "
                "low quality, bad anatomy, deformed face, disfigured, asymmetry, extra limbs, watermark, text, artifacts, "
                "overexposed, underexposed"
            )

            # --> 여기서 기존 generate_image 대신 cosmicman 함수로 교체
            generate_cosmicman_image(
                prompt + f", {acquaintance['age']} , {acquaintance['nationality or species']},{acquaintance['gender']}"
                + physical_description + "8k",
                negative_prompt,
                person_image_path
            )


##########################################################################
# 4. 메인 실행부 (예시)
##########################################################################
if __name__ == "__main__":
    data_path = '/home/chanho/Model/photo-sharing/pre_code/dataset/data_10.json'
    API_KEY = ''
    model_name = 'gpt-4o'
  
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # # 예: 메인 폴더 구조 및 스피커 얼굴 생성
    # for dialogue in data:
    #     speaker1 = dialogue['speaker1']['name'].replace("*", "").strip()
    #     speaker2 = dialogue['speaker2']['name'].replace("*", "").strip()

    #     speaker1_face = dialogue['speaker1']["physical_description"]
    #     speaker2_face = dialogue['speaker2']["physical_description"]

    #     folder_name = f"{speaker1}-{speaker2}"
    #     main_folder_path = os.path.join('./generated_images', folder_name)
    #     os.makedirs(main_folder_path, exist_ok=True)

    #     # face, speaker1, speaker2 폴더 생성
    #     face_folder_path = os.path.join(main_folder_path, 'face')
    #     speaker1_folder_path = os.path.join(main_folder_path, speaker1)
    #     speaker2_folder_path = os.path.join(main_folder_path, speaker2)

    #     os.makedirs(face_folder_path, exist_ok=True)
    #     os.makedirs(speaker1_folder_path, exist_ok=True)
    #     os.makedirs(speaker2_folder_path, exist_ok=True)

    #     # 이미지 파일은 face 폴더에 저장
    #     speaker1_image_path = os.path.join(face_folder_path, speaker1 + '.png')
    #     speaker2_image_path = os.path.join(face_folder_path, speaker2 + '.png')

    #     # 공통 프롬프트
    #     prompt = (
    #         "A realistic photograph of a human, upper body and face, natural skin texture, natural lighting, sharp focus, "
    #         "professional DSLR camera shot, high resolution, realistic clothing, studio-quality photo, taken at eye level"
    #     )
    #     negative_prompt = (
    #         "cartoon, 3D render, illustration, unrealistic, exaggerated features, low quality, blurry, distorted, artificial, "
    #         "plastic, overprocessed, painting, CGI, abstract, digital art, oversaturated, overexposed, harsh shadows, strange proportions"
    #     )

    #     # --> 스피커 1, 2 얼굴 생성에 cosmicman 사용
    #     generate_cosmicman_image(
    #         prompt + f",{dialogue['speaker1']['gender']},{dialogue['speaker1']['age']},{dialogue['speaker1']['nationality']}"
    #         + remove_prefix(speaker1_face),
    #         negative_prompt,
    #         speaker1_image_path
    #     )

    #     generate_cosmicman_image(
    #         prompt + f",{dialogue['speaker2']['gender']},{dialogue['speaker2']['age']},{dialogue['speaker2']['nationality']}"
    #         + remove_prefix(speaker2_face) + "8k",
    #         negative_prompt,
    #         speaker2_image_path
    #     )

    # # 외부인(acquaintance) 처리 - profile + face 이미지
    # print('making acquaintance')
    # search_acquaintance(data, model_name, api_key=API_KEY)
    # print("finish making acquaintance")

    # print("making face acquaintance")
    # make_face_acquaintance(data)

    # 각 대화(세션)에서 나온 landscape 이미지 생성 로직
    for dialogue in data:
        speaker1 = dialogue['speaker1']['name'].replace("*", "").strip()
        speaker2 = dialogue['speaker2']['name'].replace("*", "").strip()
        
        folder_name = f"{speaker1}-{speaker2}"
        main_folder_path = os.path.join('./generated_images', folder_name)
        speaker1_folder_path = os.path.join(main_folder_path, speaker1)
        speaker2_folder_path = os.path.join(main_folder_path, speaker2)
        face_folder_path = os.path.join(main_folder_path, 'face')

        for num in range(1, 6):
            session_dialogs = dialogue[f'session_{num}']['dialogue']
            for utter in session_dialogs:
                if 'photo_description' not in utter:
                    continue

                photo_desc = utter['photo_description']
                speaker = utter['speaker']
                date = utter['day']
                people_raw = utter['people']

                # landscape 처리
                if people_raw == 'landscape':
                    if speaker in speaker1:
                        out_path = os.path.join(speaker1_folder_path, f'{speaker1}-landscape-{date}.png')
                    else:
                        out_path = os.path.join(speaker2_folder_path, f'{speaker2}-landscape-{date}.png')

                    # landscape는 그대로 generate_landscape 사용
                    generate_landscape_image(photo_desc + " 8k", "low quality, unrealistic, cartoonish, fantasy, abstract, person ", out_path)
                    continue
            # ------------------------------
            # 1) 사람 리스트화
            # ------------------------------
                people_list = [p.strip() for p in people_raw.split(',')]
                # 'me' 키워드를 실제 speaker 이름으로 치환
                people_list = [speaker if p == 'me' else p for p in people_list]

                # 출력 경로 설정
                if speaker in speaker1:
                    folder_path = speaker1_folder_path
                else:
                    folder_path = speaker2_folder_path

                out_path = os.path.join(folder_path, f"{speaker}-people-{date}.png")

                # ------------------------------
                # 2) face 폴더에서 이미지 경로 매칭
                # ------------------------------
                face_files = os.listdir(face_folder_path)
                matched_files = []

                for person_name in people_list:
                    # 파일명(확장자 제외) 혹은 부분 문자열과 비교해 매칭
                    candidate = [
                        f for f in face_files 
                        if person_name.lower() in f.lower() or f.rsplit('.', 1)[0].lower() == person_name.lower()
                    ]
                    if candidate:
                        matched_files.append(os.path.join(face_folder_path, candidate[0]))

                if not matched_files:
                    print(f"[WARN] {people_list} 에 해당하는 얼굴 파일이 없습니다. 스킵합니다.")
                    continue

                # ------------------------------
                # 3) 사람 수에 따라 다른 모델로 T2I 수행
                # ------------------------------
                num_people = len(matched_files)
                print(speaker)
                print(people_raw)
                print(f"[INFO] 현재 사진 속 인물 수: {num_people}명")

                # ※ 1명도 CosmicMan 대신, 동일하게 UniPortrait로 생성
                if 1 <= num_people <= 5:
                    print(f" -> {num_people}명: UniPortrait multi-face 호출")
                    generate_uniportrait_image_with_multiple_faces(
                        prompt           = photo_desc + " 8k",
                        negative_prompt  = (
                            "multiple people, strange faces, low quality, distorted features, artifacts, "
                            "collage, nude, NSFW, explicit, violent, gore"
                        ),
                        face_image_paths = matched_files,
                        output_path      = out_path,
                        faceid_scale     = 0.7,
                        face_structure_scale = 0.6,
                        num_samples      = 1,
                        seed             = 42,
                        resolution       = (512, 512),
                        steps            = 80
                    )
                else:
                    print(f"[WARN] 5명을 초과하는 얼굴은 지원하지 않습니다. (감지된 인원: {num_people}명)")

    print("=== 모든 작업 완료 ===")
